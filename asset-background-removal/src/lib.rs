//! Offline, ordered InSPyReNet inference for local animation frames.
//!
//! A process-wide worker owns the Python child on a dedicated OS thread. This
//! matters because UI operations create short-lived Tokio runtimes; keeping a
//! Tokio child or pipe in a global would otherwise retain a dead reactor.
use image::{GenericImageView, GrayImage, RgbaImage};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{OnceLock, mpsc},
    thread,
    time::Duration,
};
use tokio::sync::{mpsc as async_mpsc, oneshot};
use tokio_util::sync::CancellationToken;

/// Exact embedded Python runner source, exported for compatible cache identities.
pub const RUNNER_SOURCE: &str = include_str!("../scripts/inspyrenet_video.py");
const RUNNER: &str = RUNNER_SOURCE;
const MAX_FRAMES: usize = 2_000;
const MAX_WIDTH: u32 = 3_840;
const MAX_HEIGHT: u32 = 2_160;
const COMMAND_QUEUE: usize = 1;
const SERVER_EVENT_QUEUE: usize = 4;
const MAX_REQUEST_BYTES: usize = 1_048_576;
const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Clone, serde::Serialize)]
pub struct InspyReNet {
    pub python: PathBuf,
    pub checkpoint: PathBuf,
    pub device: String,
}

/// A raw, source-sized grayscale model mask ready for CPU compositing.
#[derive(Debug, Clone)]
pub struct MaskFrame {
    pub index: usize,
    pub source: PathBuf,
    pub mask: PathBuf,
}

#[derive(Clone)]
struct FrameList {
    count: usize,
    width: u32,
    height: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RuntimeKey(String);

impl InspyReNet {
    pub fn from_environment() -> Result<Self, String> {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .ok_or("HOME is unavailable; configure the InSPyReNet runtime.")?;
        let installed = home.join(".local/share/asset-scaler/inspyrenet-venv/bin/python");
        let legacy_installed = home.join(".local/share/sprite-studio/inspyrenet-venv/bin/python");
        let runtime = Self {
            python: std::env::var_os("ASSET_SCALER_INSPYRENET_PYTHON")
                .or_else(|| std::env::var_os("SPRITE_STUDIO_INSPYRENET_PYTHON"))
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    if installed.is_file() {
                        installed
                    } else if legacy_installed.is_file() {
                        legacy_installed
                    } else {
                        PathBuf::from("python3")
                    }
                }),
            checkpoint: std::env::var_os("ASSET_SCALER_INSPYRENET_CHECKPOINT")
                .or_else(|| std::env::var_os("SPRITE_STUDIO_INSPYRENET_CHECKPOINT"))
                .map(PathBuf::from)
                .unwrap_or_else(|| home.join(".transparent-background/ckpt_base.pth")),
            device: std::env::var("ASSET_SCALER_INSPYRENET_DEVICE")
                .or_else(|_| std::env::var("SPRITE_STUDIO_INSPYRENET_DEVICE"))
                .unwrap_or_else(|_| "auto".into()),
        };
        runtime.validate()?;
        Ok(runtime)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.checkpoint.is_file() {
            return Err(format!(
                "InSPyReNet checkpoint is missing at {}. Set ASSET_SCALER_INSPYRENET_CHECKPOINT (SPRITE_STUDIO_INSPYRENET_CHECKPOINT is accepted for compatibility); this runtime never downloads models.",
                self.checkpoint.display()
            ));
        }
        if !matches!(self.device.as_str(), "auto" | "cpu" | "cuda") {
            return Err("ASSET_SCALER_INSPYRENET_DEVICE must be auto, cpu, or cuda (SPRITE_STUDIO_INSPYRENET_DEVICE is accepted for compatibility).".into());
        }
        Ok(())
    }

    /// A content-addressed identity for caches which depend on the runner,
    /// Python runtime, checkpoint, and device policy.
    pub async fn cache_identity(&self, cancellation: &CancellationToken) -> Result<String, String> {
        self.validate()?;
        self.runtime_key(cancellation).await.map(|key| key.0)
    }

    /// Starts (or reuses) the persistent model process and verifies its ready
    /// protocol. Unlike the old `--check` process, this warm-up is reusable.
    pub async fn validate_runtime(&self, cancellation: &CancellationToken) -> Result<(), String> {
        self.validate()?;
        let command_cancellation = cancellation.child_token();
        let _cleanup = CancelOnDrop(command_cancellation.clone());
        let key = self.runtime_key(&command_cancellation).await?;
        self.validate_runtime_with_key(key, &command_cancellation)
            .await
    }

    /// Verifies the persistent model process using an identity returned by
    /// [`Self::cache_identity`]. This avoids recomputing the Python runtime
    /// identity when a caller retains it for later inference.
    pub async fn validate_runtime_with_identity(
        &self,
        cancellation: &CancellationToken,
        identity: &str,
    ) -> Result<(), String> {
        self.validate()?;
        let command_cancellation = cancellation.child_token();
        let _cleanup = CancelOnDrop(command_cancellation.clone());
        self.validate_runtime_with_key(RuntimeKey(identity.to_owned()), &command_cancellation)
            .await
    }

    async fn validate_runtime_with_key(
        &self,
        key: RuntimeKey,
        cancellation: &CancellationToken,
    ) -> Result<(), String> {
        let (result, receiver) = oneshot::channel();
        submit_command(
            ServiceCommand::Validate {
                runtime: self.clone(),
                key,
                cancellation: cancellation.clone(),
                result,
            },
            cancellation,
        )
        .await?;
        await_result(receiver, "InSPyReNet runtime validation").await
    }

    #[cfg(test)]
    async fn stream_masks(
        &self,
        frames: &[PathBuf],
        directory: &Path,
        cancellation: &CancellationToken,
        sender: async_mpsc::Sender<MaskFrame>,
    ) -> Result<(), String> {
        self.validate()?;
        let key = self.runtime_key(cancellation).await?;
        self.stream_masks_with_key(frames, directory, cancellation, sender, key)
            .await
    }

    /// Streams ordered masks using an identity already computed for the
    /// enclosing processed-frame cache operation. The bounded sender applies
    /// backpressure before decoded frames accumulate in memory.
    pub async fn stream_masks_with_identity(
        &self,
        frames: &[PathBuf],
        directory: &Path,
        cancellation: &CancellationToken,
        sender: async_mpsc::Sender<MaskFrame>,
        identity: &str,
    ) -> Result<(), String> {
        self.validate()?;
        self.stream_masks_with_key(
            frames,
            directory,
            cancellation,
            sender,
            RuntimeKey(identity.to_owned()),
        )
        .await
    }

    async fn stream_masks_with_key(
        &self,
        frames: &[PathBuf],
        directory: &Path,
        cancellation: &CancellationToken,
        sender: async_mpsc::Sender<MaskFrame>,
        key: RuntimeKey,
    ) -> Result<(), String> {
        let command_cancellation = cancellation.child_token();
        let _cleanup = CancelOnDrop(command_cancellation.clone());
        let input = prepare_frames(frames, directory, &command_cancellation).await?;
        let (result, receiver) = oneshot::channel();
        submit_command(
            ServiceCommand::Infer {
                request: InferRequest {
                    runtime: self.clone(),
                    key,
                    frames: frames.to_vec(),
                    directory: directory.to_path_buf(),
                    input,
                    cancellation: command_cancellation.clone(),
                    sender,
                },
                result,
            },
            &command_cancellation,
        )
        .await?;
        await_result(receiver, "InSPyReNet mask generation").await
    }

    async fn runtime_key(&self, cancellation: &CancellationToken) -> Result<RuntimeKey, String> {
        let python = self.python.clone();
        let checkpoint = self.checkpoint.clone();
        let device = self.device.clone();
        let token = cancellation.clone();
        let base = tokio::task::spawn_blocking(move || {
            runtime_key_sync(&python, &checkpoint, &device, &token)
        })
        .await
        .map_err(|error| error.to_string())??;
        let python = self.python.clone();
        let checkpoint = self.checkpoint.clone();
        let device = self.device.clone();
        let token = cancellation.clone();
        let backend = tokio::task::spawn_blocking(move || {
            runtime_identity_sync(&python, &checkpoint, &device, &token)
        })
        .await
        .map_err(|error| error.to_string())??;
        let mut hash = Sha256::new();
        hash.update(base.0.as_bytes());
        hash.update([0]);
        hash.update(backend.as_bytes());
        Ok(RuntimeKey(format!("{:x}", hash.finalize())))
    }
}

struct CancelOnDrop(CancellationToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// Load one source and model mask, remapping the source-sized mask before
/// applying the historical integer alpha formula. A 100% feather keeps the
/// raw model confidence, while zero makes it a hard mask at confidence 0.5.
pub fn apply_mask_to_rgba_with_feather(
    source: &Path,
    mask: &Path,
    feather_percent: u8,
) -> Result<RgbaImage, String> {
    if feather_percent > 100 {
        return Err("InSPyReNet mask feather must be from 0 through 100 percent.".into());
    }
    let mut image = image::open(source)
        .map_err(|error| {
            format!(
                "InSPyReNet input frame {} is unreadable: {error}",
                source.display()
            )
        })?
        .into_rgba8();
    let mask_image = image::open(mask)
        .map_err(|error| format!("InSPyReNet mask {} is unreadable: {error}", mask.display()))?;
    if mask_image.color() != image::ColorType::L8 {
        return Err(format!(
            "InSPyReNet mask {} is not an 8-bit grayscale PNG.",
            mask.display()
        ));
    }
    apply_mask_with_feather(&mut image, &mask_image.into_luma8(), feather_percent)?;
    Ok(image)
}

async fn prepare_frames(
    frames: &[PathBuf],
    directory: &Path,
    cancellation: &CancellationToken,
) -> Result<FrameList, String> {
    if cancellation.is_cancelled() {
        return Err("InSPyReNet frame preparation cancelled.".into());
    }
    if frames.is_empty() || frames.len() > MAX_FRAMES {
        return Err(format!(
            "InSPyReNet requires 1 to {MAX_FRAMES} input frames."
        ));
    }
    tokio::fs::create_dir_all(directory)
        .await
        .map_err(|error| error.to_string())?;
    let paths = frames.to_vec();
    let token = cancellation.clone();
    let (width, height) = tokio::task::spawn_blocking(move || {
        let mut expected = None;
        for (index, path) in paths.iter().enumerate() {
            if token.is_cancelled() {
                return Err("InSPyReNet frame preparation cancelled.".to_owned());
            }
            let actual = image::image_dimensions(path).map_err(|error| {
                format!(
                    "InSPyReNet input frame {} is unreadable: {error}",
                    index + 1
                )
            })?;
            if actual.0 == 0 || actual.1 == 0 || actual.0 > MAX_WIDTH || actual.1 > MAX_HEIGHT {
                return Err(format!(
                    "InSPyReNet input frame {} has unsupported dimensions.",
                    index + 1
                ));
            }
            if let Some(size) = expected {
                if actual != size {
                    return Err(format!(
                        "InSPyReNet input frame {} is {actual:?}; expected {size:?}.",
                        index + 1
                    ));
                }
            } else {
                expected = Some(actual);
            }
        }
        expected.ok_or_else(|| "InSPyReNet requires at least one frame.".to_owned())
    })
    .await
    .map_err(|error| error.to_string())??;
    Ok(FrameList {
        count: frames.len(),
        width,
        height,
    })
}

fn runtime_key_sync(
    python: &Path,
    checkpoint: &Path,
    device: &str,
    cancellation: &CancellationToken,
) -> Result<RuntimeKey, String> {
    let mut hash = Sha256::new();
    hash.update(b"sprite-studio-inspyrenet-service-v1\0");
    hash.update(RUNNER.as_bytes());
    hash.update([0]);
    fingerprint_runtime_file(&mut hash, python, cancellation)?;
    fingerprint_file(&mut hash, checkpoint, cancellation)?;
    hash.update(device.as_bytes());
    Ok(RuntimeKey(format!("{:x}", hash.finalize())))
}

fn runtime_identity_sync(
    python: &Path,
    checkpoint: &Path,
    device: &str,
    cancellation: &CancellationToken,
) -> Result<String, String> {
    let mut child = Command::new(python)
        .args(["-c", RUNNER, "--runtime-identity", "--checkpoint"])
        .arg(checkpoint)
        .args(["--device", device])
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("Could not inspect InSPyReNet runtime: {error}"))?;
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        if cancellation.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            return Err("InSPyReNet runtime identity cancelled.".into());
        }
        if child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_some()
        {
            let output = child
                .wait_with_output()
                .map_err(|error| error.to_string())?;
            if !output.status.success() {
                return Err(format!(
                    "Could not inspect InSPyReNet runtime: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                ));
            }
            let value: serde_json::Value = serde_json::from_slice(&output.stdout)
                .map_err(|error| format!("InSPyReNet runtime identity is invalid: {error}"))?;
            return serde_json::to_string(&value).map_err(|error| error.to_string());
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err("InSPyReNet runtime identity timed out.".into());
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn fingerprint_runtime_file(
    hash: &mut Sha256,
    path: &Path,
    cancellation: &CancellationToken,
) -> Result<(), String> {
    hash.update(path.as_os_str().as_encoded_bytes());
    hash.update([0]);
    let resolved = if path.is_file() {
        Some(path.to_path_buf())
    } else {
        std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|directory| directory.join(path))
                .find(|candidate| candidate.is_file())
        })
    };
    if let Some(resolved) = resolved {
        hash.update(resolved.as_os_str().as_encoded_bytes());
        hash.update([0]);
        fingerprint_file(hash, &resolved, cancellation)?;
    }
    Ok(())
}

fn fingerprint_file(
    hash: &mut Sha256,
    path: &Path,
    cancellation: &CancellationToken,
) -> Result<(), String> {
    let mut file = File::open(path)
        .map_err(|error| format!("Could not fingerprint {}: {error}", path.display()))?;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        if cancellation.is_cancelled() {
            return Err("InSPyReNet runtime fingerprint cancelled.".into());
        }
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            return Ok(());
        }
        hash.update(&buffer[..count]);
    }
}

enum ServiceCommand {
    Validate {
        runtime: InspyReNet,
        key: RuntimeKey,
        cancellation: CancellationToken,
        result: oneshot::Sender<Result<(), String>>,
    },
    Infer {
        request: InferRequest,
        result: oneshot::Sender<Result<(), String>>,
    },
}

struct InferRequest {
    runtime: InspyReNet,
    key: RuntimeKey,
    frames: Vec<PathBuf>,
    directory: PathBuf,
    input: FrameList,
    cancellation: CancellationToken,
    sender: async_mpsc::Sender<MaskFrame>,
}

struct ServiceClient {
    commands: mpsc::SyncSender<ServiceCommand>,
}

fn service_client() -> &'static ServiceClient {
    static CLIENT: OnceLock<ServiceClient> = OnceLock::new();
    CLIENT.get_or_init(|| {
        let (sender, receiver) = mpsc::sync_channel(COMMAND_QUEUE);
        thread::Builder::new()
            .name("inspyrenet-service".into())
            .spawn(move || ServiceWorker::new(receiver).run())
            .expect("could not start InSPyReNet service thread");
        ServiceClient { commands: sender }
    })
}

/// Tests that exercise the process-wide service must serialize even when Rust
/// runs module tests in parallel. Kept crate-visible for pipeline integration
/// tests that use a fake InSPyReNet runtime.
#[cfg(any(test, feature = "test-support"))]
pub async fn test_service_lock() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

async fn submit_command(
    command: ServiceCommand,
    cancellation: &CancellationToken,
) -> Result<(), String> {
    let mut command = Some(command);
    loop {
        if cancellation.is_cancelled() {
            return Err("InSPyReNet request cancelled.".into());
        }
        let pending = command.take().expect("command is retained until queued");
        match service_client().commands.try_send(pending) {
            Ok(()) => return Ok(()),
            Err(mpsc::TrySendError::Full(pending)) => {
                command = Some(pending);
                tokio::select! {
                    _ = cancellation.cancelled() => return Err("InSPyReNet request cancelled.".into()),
                    _ = tokio::time::sleep(Duration::from_millis(5)) => {}
                }
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                return Err("InSPyReNet service stopped unexpectedly.".into());
            }
        }
    }
}

async fn await_result(
    receiver: oneshot::Receiver<Result<(), String>>,
    task: &str,
) -> Result<(), String> {
    receiver
        .await
        .map_err(|_| format!("{task} service stopped unexpectedly."))?
}

struct ServiceWorker {
    commands: mpsc::Receiver<ServiceCommand>,
    service: Option<PythonService>,
    next_request: u64,
}

impl ServiceWorker {
    fn new(commands: mpsc::Receiver<ServiceCommand>) -> Self {
        Self {
            commands,
            service: None,
            next_request: 0,
        }
    }

    fn run(mut self) {
        while let Ok(command) = self.commands.recv() {
            match command {
                ServiceCommand::Validate {
                    runtime,
                    key,
                    cancellation,
                    result,
                } => {
                    let outcome = self.ensure_service(&runtime, key, &cancellation);
                    let _ = result.send(outcome.map(|_| ()));
                }
                ServiceCommand::Infer { request, result } => {
                    let outcome = self.infer(request);
                    let _ = result.send(outcome);
                }
            }
        }
        self.stop_service();
    }

    fn ensure_service(
        &mut self,
        runtime: &InspyReNet,
        key: RuntimeKey,
        cancellation: &CancellationToken,
    ) -> Result<(), String> {
        if cancellation.is_cancelled() {
            return Err("InSPyReNet request cancelled.".into());
        }
        if self
            .service
            .as_ref()
            .is_some_and(|service| service.key == key)
        {
            if self
                .service
                .as_mut()
                .expect("service was checked")
                .exited()?
            {
                self.stop_service();
            } else {
                return Ok(());
            }
        }
        self.stop_service();
        let service = PythonService::start(runtime, key, cancellation)?;
        self.service = Some(service);
        Ok(())
    }

    fn infer(&mut self, request: InferRequest) -> Result<(), String> {
        let InferRequest {
            runtime,
            key,
            frames,
            directory,
            input,
            cancellation,
            sender,
        } = request;
        self.ensure_service(&runtime, key, &cancellation)?;
        self.next_request = self.next_request.wrapping_add(1);
        let request = self.next_request;
        let paths = frames
            .iter()
            .map(|path| path_to_utf8(path.as_path()))
            .collect::<Result<Vec<_>, _>>()?;
        let output = path_to_utf8(directory.as_path())?;
        let payload = serde_json::to_string(&ClientRequest {
            request,
            frames: paths,
            output,
        })
        .map_err(|error| error.to_string())?;
        if payload.len() > MAX_REQUEST_BYTES {
            return Err("InSPyReNet frame request is too large.".into());
        }
        let deadline =
            std::time::Instant::now() + Duration::from_secs(30 + input.count as u64 * 120);
        self.service
            .as_mut()
            .expect("service was just started")
            .send(&payload, &cancellation, deadline)
            .map_err(|error| self.invalidate(error))?;
        let mut complete = 0_usize;
        loop {
            if cancellation.is_cancelled() {
                return Err(self.invalidate("InSPyReNet mask generation cancelled.".into()));
            }
            if std::time::Instant::now() >= deadline {
                return Err(self.invalidate("InSPyReNet mask generation timed out.".into()));
            }
            let event = match self
                .service
                .as_ref()
                .expect("service is active")
                .next_event()
            {
                Ok(event) => event,
                Err(error) => return Err(self.invalidate(error)),
            };
            match event {
                Some(ServerMessage::Mask {
                    request: actual,
                    index,
                }) if actual == request => {
                    if index != complete || index >= input.count {
                        return Err(
                            self.invalidate("InSPyReNet emitted an invalid mask order.".into())
                        );
                    }
                    let mask = directory.join("masks").join(format!("mask-{index:06}.png"));
                    if let Err(error) = validate_mask(&mask, input.width, input.height) {
                        return Err(self.invalidate(error));
                    }
                    complete += 1;
                    let event = MaskFrame {
                        index,
                        source: frames[index].clone(),
                        mask,
                    };
                    if let Err(error) = send_mask(sender.clone(), event, &cancellation, deadline) {
                        return Err(self.invalidate(error));
                    }
                    if let Err(error) = self
                        .service
                        .as_mut()
                        .expect("service is active")
                        .acknowledge(request, index, &cancellation, deadline)
                    {
                        return Err(self.invalidate(error));
                    }
                }
                Some(ServerMessage::Complete {
                    request: actual,
                    frame_count,
                    width,
                    height,
                }) if actual == request => {
                    if (frame_count, width, height) != (input.count, input.width, input.height)
                        || complete != input.count
                    {
                        return Err(self.invalidate(
                            "InSPyReNet completion does not match approved source frames.".into(),
                        ));
                    }
                    return Ok(());
                }
                Some(ServerMessage::Error {
                    request: actual,
                    message,
                }) if actual == request => {
                    return Err(
                        self.invalidate(format!("InSPyReNet mask generation failed: {message}"))
                    );
                }
                Some(ServerMessage::Ready) => {
                    return Err(self.invalidate("InSPyReNet repeated its ready event.".into()));
                }
                Some(_) => {
                    return Err(
                        self.invalidate("InSPyReNet emitted an unexpected protocol event.".into())
                    );
                }
                None => match self.service.as_mut().expect("service is active").exited() {
                    Ok(true) => {
                        return Err(self.invalidate(
                            "InSPyReNet service stopped before completing inference.".into(),
                        ));
                    }
                    Ok(false) => {}
                    Err(error) => return Err(self.invalidate(error)),
                },
            }
        }
    }

    fn invalidate(&mut self, message: String) -> String {
        let log = self
            .service
            .as_ref()
            .map(PythonService::log_tail)
            .unwrap_or_default();
        self.stop_service();
        if log.is_empty() {
            message
        } else {
            format!("{message} Log: {log}")
        }
    }

    fn stop_service(&mut self) {
        if let Some(mut service) = self.service.take() {
            service.stop();
        }
    }
}

#[derive(Serialize)]
struct ClientRequest {
    request: u64,
    frames: Vec<String>,
    output: String,
}

#[derive(Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum ServerMessage {
    Ready,
    Mask {
        request: u64,
        index: usize,
    },
    Complete {
        request: u64,
        frame_count: usize,
        width: u32,
        height: u32,
    },
    Error {
        request: u64,
        message: String,
    },
}

struct PythonService {
    key: RuntimeKey,
    child: Child,
    events: mpsc::Receiver<Result<ServerMessage, String>>,
    reader: Option<thread::JoinHandle<()>>,
    writes: Option<mpsc::SyncSender<WriterCommand>>,
    writer: Option<thread::JoinHandle<()>>,
    log: tempfile::NamedTempFile,
}

impl PythonService {
    fn start(
        runtime: &InspyReNet,
        key: RuntimeKey,
        cancellation: &CancellationToken,
    ) -> Result<Self, String> {
        let log = tempfile::NamedTempFile::new().map_err(|error| error.to_string())?;
        let stderr = log.reopen().map_err(|error| error.to_string())?;
        let mut child = Command::new(&runtime.python)
            .args(["-c", RUNNER, "--server", "--checkpoint"])
            .arg(&runtime.checkpoint)
            .args([
                "--device",
                &runtime.device,
                "--max-frames",
                &MAX_FRAMES.to_string(),
                "--max-width",
                &MAX_WIDTH.to_string(),
                "--max-height",
                &MAX_HEIGHT.to_string(),
            ])
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr)
            .spawn()
            .map_err(|error| {
                format!(
                    "Could not launch InSPyReNet Python {}: {error}",
                    runtime.python.display()
                )
            })?;
        let stdin = child
            .stdin
            .take()
            .ok_or("InSPyReNet Python has no stdin.")?;
        let stdout = child
            .stdout
            .take()
            .ok_or("InSPyReNet Python has no stdout.")?;
        let (writes, write_commands) = mpsc::sync_channel(COMMAND_QUEUE);
        let writer = match thread::Builder::new()
            .name("inspyrenet-writer".into())
            .spawn(move || write_protocol(stdin, write_commands))
        {
            Ok(writer) => writer,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error.to_string());
            }
        };
        let (sender, events) = mpsc::sync_channel(SERVER_EVENT_QUEUE);
        let reader = match thread::Builder::new()
            .name("inspyrenet-protocol".into())
            .spawn(move || read_protocol(stdout, sender))
        {
            Ok(reader) => reader,
            Err(error) => {
                drop(writes);
                let _ = child.kill();
                let _ = child.wait();
                let _ = writer.join();
                return Err(error.to_string());
            }
        };
        let mut service = Self {
            key,
            child,
            events,
            reader: Some(reader),
            writes: Some(writes),
            writer: Some(writer),
            log,
        };
        match service.wait_ready(cancellation) {
            Ok(()) => Ok(service),
            Err(error) => {
                service.stop();
                Err(error)
            }
        }
    }

    fn wait_ready(&mut self, cancellation: &CancellationToken) -> Result<(), String> {
        let deadline = std::time::Instant::now() + STARTUP_TIMEOUT;
        loop {
            if cancellation.is_cancelled() {
                return Err("InSPyReNet runtime validation cancelled.".into());
            }
            if std::time::Instant::now() >= deadline {
                return Err("InSPyReNet runtime validation timed out.".into());
            }
            match self.next_event()? {
                Some(ServerMessage::Ready) => return Ok(()),
                Some(ServerMessage::Error { message, .. }) => {
                    return Err(format!("InSPyReNet runtime validation failed: {message}"));
                }
                Some(_) => return Err("InSPyReNet emitted an unexpected startup event.".into()),
                None if self.exited()? => {
                    return Err(format!(
                        "InSPyReNet runtime validation failed: {}",
                        self.log_tail()
                    ));
                }
                None => {}
            }
        }
    }

    fn send(
        &mut self,
        line: &str,
        cancellation: &CancellationToken,
        deadline: std::time::Instant,
    ) -> Result<(), String> {
        let (result, receiver) = mpsc::channel();
        let mut command = Some(WriterCommand {
            line: format!("{line}\n").into_bytes(),
            result,
        });
        loop {
            if cancellation.is_cancelled() {
                return Err("InSPyReNet mask generation cancelled.".into());
            }
            if std::time::Instant::now() >= deadline {
                return Err("InSPyReNet mask generation timed out while writing to Python.".into());
            }
            let writes = self
                .writes
                .as_ref()
                .ok_or("InSPyReNet Python writer stopped.")?;
            match writes.try_send(command.take().expect("write command is retained")) {
                Ok(()) => break,
                Err(mpsc::TrySendError::Full(pending)) => {
                    command = Some(pending);
                    thread::sleep(Duration::from_millis(2));
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    return Err("InSPyReNet Python writer stopped.".into());
                }
            }
        }
        loop {
            if cancellation.is_cancelled() {
                return Err("InSPyReNet mask generation cancelled.".into());
            }
            if std::time::Instant::now() >= deadline {
                return Err("InSPyReNet mask generation timed out while writing to Python.".into());
            }
            match receiver.recv_timeout(Duration::from_millis(10)) {
                Ok(result) => return result,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("InSPyReNet Python writer stopped.".into());
                }
            }
        }
    }

    fn acknowledge(
        &mut self,
        request: u64,
        index: usize,
        cancellation: &CancellationToken,
        deadline: std::time::Instant,
    ) -> Result<(), String> {
        self.send(
            &serde_json::to_string(&serde_json::json!({
                "event": "ack",
                "request": request,
                "index": index,
            }))
            .map_err(|error| error.to_string())?,
            cancellation,
            deadline,
        )
    }

    fn next_event(&self) -> Result<Option<ServerMessage>, String> {
        match self.events.recv_timeout(Duration::from_millis(25)) {
            Ok(result) => result.map(Some),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err("InSPyReNet protocol stream closed.".into())
            }
        }
    }

    fn exited(&mut self) -> Result<bool, String> {
        self.child
            .try_wait()
            .map(|status| status.is_some())
            .map_err(|error| error.to_string())
    }

    fn log_tail(&self) -> String {
        (|| -> std::io::Result<String> {
            let mut file = File::open(self.log.path())?;
            let size = file.metadata()?.len();
            let mut bytes = Vec::new();
            std::io::Seek::seek(
                &mut file,
                std::io::SeekFrom::Start(size.saturating_sub(4096)),
            )?;
            file.take(4096).read_to_end(&mut bytes)?;
            Ok(String::from_utf8_lossy(&bytes).trim().to_owned())
        })()
        .unwrap_or_else(|_| "Could not read inference log".into())
    }

    fn stop(&mut self) {
        // A blocked protocol reader may be waiting to send to this bounded
        // channel. Dropping the receiver wakes it before joining the reader.
        let (_, replacement) = mpsc::sync_channel(0);
        let events = std::mem::replace(&mut self.events, replacement);
        drop(events);
        // Drop the command sender before killing the child. A blocked pipe
        // write wakes when the child exits, and queued writes are discarded.
        drop(self.writes.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

impl Drop for PythonService {
    fn drop(&mut self) {
        self.stop();
    }
}

struct WriterCommand {
    line: Vec<u8>,
    result: mpsc::Sender<Result<(), String>>,
}

fn write_protocol(mut stdin: ChildStdin, commands: mpsc::Receiver<WriterCommand>) {
    while let Ok(command) = commands.recv() {
        let result = stdin
            .write_all(&command.line)
            .and_then(|()| stdin.flush())
            .map_err(|error| error.to_string());
        let _ = command.result.send(result);
    }
}

fn read_protocol(
    stdout: std::process::ChildStdout,
    sender: mpsc::SyncSender<Result<ServerMessage, String>>,
) {
    for line in BufReader::new(stdout).lines() {
        let message = line.map_err(|error| error.to_string()).and_then(|line| {
            serde_json::from_str(&line)
                .map_err(|error| format!("InSPyReNet protocol is invalid: {error}"))
        });
        if sender.send(message).is_err() {
            return;
        }
    }
}

fn send_mask(
    sender: async_mpsc::Sender<MaskFrame>,
    event: MaskFrame,
    cancellation: &CancellationToken,
    deadline: std::time::Instant,
) -> Result<(), String> {
    let mut event = Some(event);
    loop {
        if cancellation.is_cancelled() {
            return Err("InSPyReNet mask generation cancelled.".into());
        }
        if std::time::Instant::now() >= deadline {
            return Err("InSPyReNet mask generation timed out waiting for its consumer.".into());
        }
        match sender.try_send(
            event
                .take()
                .expect("mask event is retained until delivered"),
        ) {
            Ok(()) => return Ok(()),
            Err(async_mpsc::error::TrySendError::Full(value)) => {
                event = Some(value);
                thread::sleep(Duration::from_millis(2));
            }
            Err(async_mpsc::error::TrySendError::Closed(_)) => {
                return Err("InSPyReNet mask receiver closed.".into());
            }
        }
    }
}

fn validate_mask(path: &Path, width: u32, height: u32) -> Result<(), String> {
    let mask = image::open(path)
        .map_err(|error| format!("InSPyReNet mask {} is unreadable: {error}", path.display()))?;
    if mask.color() != image::ColorType::L8 || mask.dimensions() != (width, height) {
        return Err(format!(
            "InSPyReNet mask {} is not an 8-bit source-sized grayscale PNG.",
            path.display()
        ));
    }
    Ok(())
}

fn path_to_utf8(path: &Path) -> Result<String, String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| "InSPyReNet frame path is not valid UTF-8.".into())
}

fn apply_mask_with_feather(
    image: &mut RgbaImage,
    mask: &GrayImage,
    feather_percent: u8,
) -> Result<(), String> {
    if feather_percent > 100 {
        return Err("InSPyReNet mask feather must be from 0 through 100 percent.".into());
    }
    if image.dimensions() != mask.dimensions() {
        return Err(format!(
            "InSPyReNet mask size {:?} differs from frame size {:?}.",
            mask.dimensions(),
            image.dimensions()
        ));
    }
    for (pixel, coverage) in image.pixels_mut().zip(mask.pixels()) {
        let coverage = feathered_coverage(coverage[0], feather_percent);
        pixel[3] = ((u16::from(pixel[3]) * u16::from(coverage) + 127) / 255) as u8;
        if pixel[3] == 0 {
            pixel.0 = [0; 4];
        }
    }
    Ok(())
}

/// Maps a raw model confidence onto a symmetric linear band around 0.5. The
/// integer form rounds to the nearest alpha byte and keeps 100% bit-exact.
fn feathered_coverage(raw: u8, feather_percent: u8) -> u8 {
    match feather_percent {
        0 => u8::from(raw >= 128) * 255,
        100 => raw,
        band => {
            let numerator = 200 * i32::from(raw) - 255 * (100 - i32::from(band));
            let denominator = 2 * i32::from(band);
            if numerator <= 0 {
                0
            } else if numerator >= 255 * denominator {
                255
            } else {
                ((numerator + denominator / 2) / denominator) as u8
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgba};
    use std::os::unix::fs::PermissionsExt;

    fn fake_runtime() -> (tempfile::TempDir, InspyReNet) {
        let directory = tempfile::tempdir().unwrap();
        let python = directory.path().join("fake-inspyrenet.py");
        std::fs::write(
            &python,
            r#"#!/usr/bin/env python3
import json, os, shutil, sys
if "--runtime-identity" in sys.argv:
    print('{"backend":"cpu","torch":"fake","hip":null}', flush=True)
    raise SystemExit(0)
with open(sys.argv[0] + ".starts", "a", encoding="utf-8") as log:
    log.write("start\n")
print('{"event":"ready","runtime":{"backend":"cpu","torch":"fake","hip":null}}', flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if "error" in request["output"]:
        print(json.dumps({"event":"error","request":request["request"],"message":"fake failure"}), flush=True)
        continue
    dimensions = None
    for index, source in enumerate(request["frames"]):
        target_dir = os.path.join(request["output"], "masks")
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, "mask-%06d.png" % index)
        shutil.copyfile(source, target)
        print(json.dumps({"event":"mask","request":request["request"],"index":index}), flush=True)
        if json.loads(sys.stdin.readline()) != {"event":"ack","request":request["request"],"index":index}:
            raise SystemExit("missing ack")
        if dimensions is None:
            from struct import unpack
            with open(source, "rb") as image:
                header = image.read(24)
            dimensions = unpack(">II", header[16:24])
    print(json.dumps({"event":"complete","request":request["request"],"frame_count":len(request["frames"]),"width":dimensions[0],"height":dimensions[1]}), flush=True)
"#,
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&python).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&python, permissions).unwrap();
        let checkpoint = directory.path().join("checkpoint");
        std::fs::write(&checkpoint, b"fake model").unwrap();
        (
            directory,
            InspyReNet {
                python,
                checkpoint,
                device: "cpu".into(),
            },
        )
    }

    fn gray_source(directory: &Path, index: usize) -> PathBuf {
        let source = directory.join(format!("source-{index}.png"));
        GrayImage::from_pixel(1, 1, Luma([128]))
            .save(&source)
            .unwrap();
        source
    }

    #[test]
    fn raw_model_mask_preserves_color_and_multiplies_existing_alpha() {
        // Pure test: no process-wide lock is needed.
        let mut image = RgbaImage::from_pixel(3, 1, Rgba([10, 80, 150, 128]));
        let mask = GrayImage::from_fn(3, 1, |x, _| {
            Luma([match x {
                0 => 0,
                1 => 128,
                _ => 255,
            }])
        });
        apply_mask_with_feather(&mut image, &mask, 100).unwrap();
        assert_eq!(image.get_pixel(0, 0).0, [0; 4]);
        assert_eq!(image.get_pixel(1, 0).0, [10, 80, 150, 64]);
        assert_eq!(image.get_pixel(2, 0).0, [10, 80, 150, 128]);
        assert!(apply_mask_with_feather(&mut image, &GrayImage::new(2, 2), 100).is_err());
    }

    #[test]
    fn mask_feather_remaps_confidence_before_preserving_source_alpha() {
        let mask = GrayImage::from_fn(5, 1, |x, _| {
            Luma([match x {
                0 => 127,
                1 => 128,
                2 => 96,
                3 => 159,
                _ => 255,
            }])
        });

        let mut hard = RgbaImage::from_pixel(5, 1, Rgba([10, 80, 150, 128]));
        apply_mask_with_feather(&mut hard, &mask, 0).unwrap();
        assert_eq!(hard.get_pixel(0, 0).0, [0; 4]);
        assert_eq!(hard.get_pixel(1, 0).0, [10, 80, 150, 128]);

        let mut raw = RgbaImage::from_pixel(5, 1, Rgba([10, 80, 150, 128]));
        apply_mask_with_feather(&mut raw, &mask, 100).unwrap();
        assert_eq!(raw.get_pixel(0, 0).0, [10, 80, 150, 64]);
        assert_eq!(raw.get_pixel(1, 0).0, [10, 80, 150, 64]);
        assert_eq!(raw.get_pixel(4, 0).0, [10, 80, 150, 128]);

        let mut intermediate = RgbaImage::from_pixel(5, 1, Rgba([10, 80, 150, 128]));
        apply_mask_with_feather(&mut intermediate, &mask, 50).unwrap();
        assert!(intermediate.get_pixel(2, 0)[3] < raw.get_pixel(2, 0)[3]);
        assert!(intermediate.get_pixel(3, 0)[3] > raw.get_pixel(3, 0)[3]);
        assert_eq!(intermediate.get_pixel(4, 0).0, [10, 80, 150, 128]);
    }

    #[test]
    fn runtime_key_changes_with_checkpoint_content() {
        let directory = tempfile::tempdir().unwrap();
        let checkpoint = directory.path().join("checkpoint");
        std::fs::write(&checkpoint, b"first").unwrap();
        let token = CancellationToken::new();
        let first = runtime_key_sync(Path::new("python3"), &checkpoint, "cpu", &token).unwrap();
        std::fs::write(&checkpoint, b"second").unwrap();
        let second = runtime_key_sync(Path::new("python3"), &checkpoint, "cpu", &token).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn service_survives_two_short_lived_tokio_runtimes() {
        let (directory, runtime) = fake_runtime();
        let lock_runtime = tokio::runtime::Runtime::new().unwrap();
        let _service_lock = lock_runtime.block_on(test_service_lock());
        for _ in 0..2 {
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(runtime.validate_runtime(&CancellationToken::new()))
                .unwrap();
        }
        let starts =
            std::fs::read_to_string(format!("{}.starts", runtime.python.display())).unwrap();
        assert_eq!(starts.lines().count(), 1, "the Python model was not reused");
        drop(directory);
    }

    #[tokio::test]
    async fn stream_masks_orders_events_and_restarts_after_an_error() {
        let _service_lock = test_service_lock().await;
        let (_directory, runtime) = fake_runtime();
        let input = tempfile::tempdir().unwrap();
        let frames = (0..3)
            .map(|index| gray_source(input.path(), index))
            .collect::<Vec<_>>();
        let output = input.path().join("normal");
        let (sender, mut receiver) = async_mpsc::channel(1);
        let cancellation = CancellationToken::new();
        let task = runtime.stream_masks(&frames, &output, &cancellation, sender);
        tokio::pin!(task);
        let mut indexes = Vec::new();
        let mut receiver_closed = false;
        loop {
            tokio::select! {
                result = &mut task => {
                    result.unwrap();
                    while let Some(event) = receiver.recv().await {
                        indexes.push(event.index);
                    }
                    break;
                }
                event = receiver.recv(), if !receiver_closed => match event {
                    Some(event) => indexes.push(event.index),
                    None => receiver_closed = true,
                },
            }
        }
        assert_eq!(indexes, vec![0, 1, 2]);

        let (sender, _receiver) = async_mpsc::channel(1);
        let error_output = input.path().join("error");
        let error_cancellation = CancellationToken::new();
        let error = runtime
            .stream_masks(&frames[..1], &error_output, &error_cancellation, sender)
            .await
            .unwrap_err();
        assert!(error.contains("fake failure"));
        runtime
            .validate_runtime(&CancellationToken::new())
            .await
            .unwrap();
        let starts =
            std::fs::read_to_string(format!("{}.starts", runtime.python.display())).unwrap();
        assert_eq!(
            starts.lines().count(),
            2,
            "failure must invalidate the child"
        );
    }

    #[tokio::test]
    async fn cancellation_reaps_a_backpressured_request() {
        let _service_lock = test_service_lock().await;
        let (_directory, runtime) = fake_runtime();
        let input = tempfile::tempdir().unwrap();
        let frames = (0..3)
            .map(|index| gray_source(input.path(), index))
            .collect::<Vec<_>>();
        let cancellation = CancellationToken::new();
        let (sender, mut receiver) = async_mpsc::channel(1);
        let output = input.path().join("cancel");
        let task = runtime.stream_masks(&frames, &output, &cancellation, sender);
        tokio::pin!(task);
        // Receiving one event allows the fake server to continue until the
        // bounded sender fills again; cancellation then exercises that path.
        let first = tokio::select! {
            result = &mut task => panic!("stream ended before first mask: {result:?}"),
            event = receiver.recv() => event.unwrap(),
        };
        assert_eq!(first.index, 0);
        let third_mask = output.join("masks/mask-000002.png");
        while !third_mask.is_file() {
            tokio::select! {
                result = &mut task => panic!("stream ended before bounded backpressure: {result:?}"),
                _ = tokio::time::sleep(Duration::from_millis(2)) => {}
            }
        }
        cancellation.cancel();
        assert!(task.await.unwrap_err().contains("cancelled"));
        runtime
            .validate_runtime(&CancellationToken::new())
            .await
            .unwrap();
        let starts =
            std::fs::read_to_string(format!("{}.starts", runtime.python.display())).unwrap();
        assert_eq!(
            starts.lines().count(),
            2,
            "cancellation must invalidate the child"
        );
    }
}
