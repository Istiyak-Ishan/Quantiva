//! backend_manager.rs
//! ─────────────────────────────────────────────────────────────────────────────
//! Manages the lifecycle of the local Quantiva desktop backend child process:
//! - Ephemeral port allocation (127.0.0.1:0)
//! - Cryptographic 32-byte hex IPC token generation
//! - Headless child process spawning (CREATE_NO_WINDOW on Windows)
//! - Health check polling (GET /api/v1/system/health every 250ms up to 30s)
//! - Real-time startup progress / error event emission
//! - Graceful shutdown and process tree termination

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

// ── Types & Payloads ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
pub struct BackendConnectionInfo {
    pub base_url: String,
    pub ipc_token: String,
    pub status: String,
}

impl serde::Serialize for BackendConnectionInfo {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(5))?;
        map.serialize_entry("baseUrl", &self.base_url)?;
        map.serialize_entry("base_url", &self.base_url)?;
        map.serialize_entry("ipcToken", &self.ipc_token)?;
        map.serialize_entry("ipc_token", &self.ipc_token)?;
        map.serialize_entry("status", &self.status)?;
        map.end()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StartupProgressPayload {
    pub stage: String,
    pub message: String,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StartupErrorPayload {
    pub error: String,
    pub details: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
struct HealthResponse {
    status: Option<String>,
    app_version: Option<String>,
    startup_status: Option<String>,
    sqlite_status: Option<String>,
    vector_store_status: Option<String>,
}

// ── Internal State ───────────────────────────────────────────────────────────

struct InnerState {
    port: u16,
    ipc_token: String,
    base_url: String,
    status: String,
    child: Option<std::process::Child>,
}

pub struct BackendManager {
    state: Mutex<InnerState>,
}

impl Default for BackendManager {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendManager {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(InnerState {
                port: 0,
                ipc_token: String::new(),
                base_url: String::new(),
                status: "stopped".to_string(),
                child: None,
            }),
        }
    }

    /// Returns the current backend connection info.
    pub fn get_connection_info(&self) -> BackendConnectionInfo {
        let state = self.state.lock().unwrap();
        BackendConnectionInfo {
            base_url: state.base_url.clone(),
            ipc_token: state.ipc_token.clone(),
            status: state.status.clone(),
        }
    }

    /// Spawns the child process and polls until healthy or timeout.
    pub async fn start(&self, app: &AppHandle) -> Result<BackendConnectionInfo, String> {
        // 1. Allocate ephemeral port and IPC token.
        let port = allocate_ephemeral_port()?;
        let ipc_token = generate_ipc_token();
        let base_url = format!("http://127.0.0.1:{}", port);

        // Update state to "starting"
        {
            let mut state = self.state.lock().unwrap();
            state.port = port;
            state.ipc_token = ipc_token.clone();
            state.base_url = base_url.clone();
            state.status = "starting".to_string();
        }

        let _ = app.emit(
            "backend-startup-progress",
            StartupProgressPayload {
                stage: "init".to_string(),
                message: "Starting Quantiva desktop backend...".to_string(),
                elapsed_ms: 0,
            },
        );

        // 2. Resolve backend binary or Python script.
        let (binary, prefix_args) = resolve_backend_command()?;

        // 3. Spawn child process without console flash on Windows.
        let parent_pid = std::process::id();
        let mut cmd = std::process::Command::new(&binary);
        for arg in &prefix_args {
            cmd.arg(arg);
        }
        cmd.arg("--port")
            .arg(port.to_string())
            .arg("--token")
            .arg(&ipc_token)
            .arg("--parent-pid")
            .arg(parent_pid.to_string());

        cmd.env("QUANTIVA_IPC_TOKEN", &ipc_token);
        cmd.env("PYTHONIOENCODING", "utf-8");

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }

        let child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn backend process: {}", e))?;

        // Store child process handle
        {
            let mut state = self.state.lock().unwrap();
            state.child = Some(child);
        }

        // 4. Poll health check endpoint every 250ms (up to 30s)
        let health_url = format!("{}/api/v1/system/health", base_url);
        let start_time = Instant::now();
        let timeout = Duration::from_secs(30);
        let poll_interval = Duration::from_millis(250);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(1500))
            .build()
            .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

        let mut last_stage_emitted = String::new();

        while start_time.elapsed() < timeout {
            // Check if child exited prematurely
            {
                let mut state = self.state.lock().unwrap();
                if let Some(ref mut child) = state.child {
                    if let Ok(Some(exit_status)) = child.try_wait() {
                        let err_msg = format!(
                            "Backend process terminated prematurely with status: {}",
                            exit_status
                        );
                        state.status = "error".to_string();
                        let _ = app.emit(
                            "backend-startup-error",
                            StartupErrorPayload {
                                error: err_msg.clone(),
                                details: None,
                            },
                        );
                        return Err(err_msg);
                    }
                }
            }

            let elapsed_ms = start_time.elapsed().as_millis() as u64;

            // Emit contextual progress events as time progresses
            let (stage, msg) = if elapsed_ms < 1500 {
                ("database", "Initializing SQLite database & WAL journal...")
            } else if elapsed_ms < 4000 {
                ("storage", "Loading storage & vector stores...")
            } else {
                ("providers", "Detecting AI providers & hardware capabilities...")
            };

            if last_stage_emitted != stage {
                last_stage_emitted = stage.to_string();
                let _ = app.emit(
                    "backend-startup-progress",
                    StartupProgressPayload {
                        stage: stage.to_string(),
                        message: msg.to_string(),
                        elapsed_ms,
                    },
                );
            }

            // Health poll
            match client.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    let _health: Option<HealthResponse> = resp.json().await.ok();
                    let final_info = BackendConnectionInfo {
                        base_url: base_url.clone(),
                        ipc_token: ipc_token.clone(),
                        status: "ready".to_string(),
                    };

                    {
                        let mut state = self.state.lock().unwrap();
                        state.status = "ready".to_string();
                    }

                    let _ = app.emit(
                        "backend-startup-progress",
                        StartupProgressPayload {
                            stage: "ready".to_string(),
                            message: "Quantiva backend initialized successfully".to_string(),
                            elapsed_ms: start_time.elapsed().as_millis() as u64,
                        },
                    );

                    let _ = app.emit("backend-ready", &final_info);
                    return Ok(final_info);
                }
                _ => {
                    tokio::time::sleep(poll_interval).await;
                }
            }
        }

        // Timeout reached
        let timeout_err = "Backend health check timed out after 30 seconds".to_string();
        {
            let mut state = self.state.lock().unwrap();
            state.status = "error".to_string();
        }

        let _ = app.emit(
            "backend-startup-error",
            StartupErrorPayload {
                error: timeout_err.clone(),
                details: Some("Ensure port is free and dependencies are available".to_string()),
            },
        );

        Err(timeout_err)
    }

    /// Restarts the child process (e.g., when changing root data directory).
    pub async fn restart(&self, app: &AppHandle) -> Result<BackendConnectionInfo, String> {
        let _ = app.emit(
            "backend-startup-progress",
            StartupProgressPayload {
                stage: "restart".to_string(),
                message: "Restarting Quantiva backend...".to_string(),
                elapsed_ms: 0,
            },
        );

        self.shutdown_sync();
        tokio::time::sleep(Duration::from_millis(500)).await;
        self.start(app).await
    }

    /// Synchronous shutdown hook for app exit.
    pub fn shutdown_sync(&self) {
        let mut state = self.state.lock().unwrap();
        state.status = "shutting_down".to_string();

        if state.port > 0 && !state.ipc_token.is_empty() {
            // Attempt clean HTTP shutdown call
            let shutdown_url = format!("127.0.0.1:{}", state.port);
            if let Ok(mut stream) = std::net::TcpStream::connect_timeout(
                &shutdown_url.parse().unwrap_or_else(|_| "127.0.0.1:80".parse().unwrap()),
                Duration::from_millis(500),
            ) {
                use std::io::Write;
                let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
                let req = format!(
                    "POST /api/v1/system/shutdown HTTP/1.1\r\nHost: {}\r\nX-Quantiva-Token: {}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    shutdown_url, state.ipc_token
                );
                let _ = stream.write_all(req.as_bytes());
            }
        }

        if let Some(mut child) = state.child.take() {
            let pid = child.id();
            // Wait up to 1.5 seconds for graceful exit
            let wait_start = Instant::now();
            let mut exited = false;
            while wait_start.elapsed() < Duration::from_millis(1500) {
                if let Ok(Some(_)) = child.try_wait() {
                    exited = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }

            if !exited {
                #[cfg(target_os = "windows")]
                {
                    use std::os::windows::process::CommandExt;
                    let _ = std::process::Command::new("taskkill")
                        .args(["/F", "/T", "/PID", &pid.to_string()])
                        .creation_flags(0x08000000)
                        .output();
                }

                let _ = child.kill();
                let _ = child.wait();
            }
        }

        state.status = "stopped".to_string();
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

pub fn allocate_ephemeral_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("Failed to bind TCP listener on 127.0.0.1:0: {}", e))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("Failed to read socket address: {}", e))?
        .port();
    drop(listener);
    Ok(port)
}

pub fn generate_ipc_token() -> String {
    use rand::RngCore;
    let mut token_bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut token_bytes);
    token_bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

pub fn resolve_backend_command() -> Result<(PathBuf, Vec<String>), String> {
    // 1. Explicit override
    if let Ok(path) = std::env::var("QUANTIVA_BACKEND_PATH") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Ok((p, Vec::new()));
        }
    }

    // 2. Next to current executable (installed desktop app or sidecar bundle)
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(exe_dir) = current_exe.parent() {
            let candidates = [
                exe_dir.join("quantiva-backend.exe"),
                exe_dir.join("quantiva-backend").join("quantiva-backend.exe"),
                exe_dir.join("bin").join("quantiva-backend.exe"),
                exe_dir.join("resources").join("quantiva-backend.exe"),
            ];
            for candidate in candidates {
                if candidate.exists() {
                    return Ok((candidate, Vec::new()));
                }
            }
        }
    }

    // 3. Dev candidates relative to workspace root
    let dev_candidates = [
        PathBuf::from("backend/dist/quantiva-backend/quantiva-backend.exe"),
        PathBuf::from("../backend/dist/quantiva-backend/quantiva-backend.exe"),
        PathBuf::from("../../backend/dist/quantiva-backend/quantiva-backend.exe"),
    ];
    for candidate in dev_candidates {
        if candidate.exists() {
            return Ok((candidate, Vec::new()));
        }
    }

    // 4. Fallback in dev: run python run_desktop_backend.py
    let py_scripts = [
        PathBuf::from("backend/run_desktop_backend.py"),
        PathBuf::from("../backend/run_desktop_backend.py"),
        PathBuf::from("../../backend/run_desktop_backend.py"),
    ];
    for script in py_scripts {
        if script.exists() {
            let python_bin = if cfg!(windows) { "python" } else { "python3" };
            return Ok((
                PathBuf::from(python_bin),
                vec![script.to_string_lossy().to_string()],
            ));
        }
    }

    Err("Could not locate quantiva-backend.exe or run_desktop_backend.py".to_string())
}

pub fn open_data_folder(subfolder: Option<String>) -> Result<(), String> {
    let data_dir = if let Ok(val) = std::env::var("QUANTIVA_DATA_DIR") {
        PathBuf::from(val)
    } else if let Ok(val) = std::env::var("LOCALAPPDATA") {
        PathBuf::from(val).join("Quantiva")
    } else {
        PathBuf::from(".").join("data")
    };

    let target = match subfolder.as_deref() {
        Some("logs") => data_dir.join("logs"),
        Some(s) if !s.trim().is_empty() => data_dir.join(s),
        _ => data_dir,
    };

    if !target.exists() {
        let _ = std::fs::create_dir_all(&target);
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        std::process::Command::new("explorer")
            .arg(&target)
            .creation_flags(0x08000000)
            .spawn()
            .map_err(|e| format!("Failed to open Windows Explorer: {}", e))?;
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::process::Command::new("open")
            .arg(&target)
            .spawn()
            .map_err(|e| format!("Failed to open file manager: {}", e))?;
    }

    Ok(())
}
