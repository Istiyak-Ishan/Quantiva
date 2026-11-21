pub mod backend_manager;

use backend_manager::{BackendConnectionInfo, BackendManager};
use std::sync::Arc;
use tauri::Manager;

#[tauri::command]
fn get_backend_connection_info(
    state: tauri::State<'_, Arc<BackendManager>>,
) -> BackendConnectionInfo {
    state.get_connection_info()
}

#[tauri::command]
async fn restart_backend(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<BackendManager>>,
) -> Result<BackendConnectionInfo, String> {
    state.restart(&app).await
}

#[tauri::command]
async fn open_data_folder(subfolder: Option<String>) -> Result<(), String> {
    backend_manager::open_data_folder(subfolder)
}

pub fn run() {
    let backend_manager = Arc::new(BackendManager::new());
    let backend_manager_for_setup = Arc::clone(&backend_manager);
    let backend_manager_for_exit = Arc::clone(&backend_manager);

    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(backend_manager)
        .setup(move |app| {
            let app_handle = app.handle().clone();
            let mgr = Arc::clone(&backend_manager_for_setup);
            tauri::async_runtime::spawn(async move {
                if let Err(e) = mgr.start(&app_handle).await {
                    eprintln!("[BackendManager] Failed to start backend: {}", e);
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_backend_connection_info,
            restart_backend,
            open_data_folder,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(move |_app_handle, event| {
            match event {
                tauri::RunEvent::ExitRequested { .. } => {
                    backend_manager_for_exit.shutdown_sync();
                }
                tauri::RunEvent::WindowEvent {
                    event: tauri::WindowEvent::CloseRequested { .. },
                    ..
                } => {
                    // Window close requested
                }
                _ => {}
            }
        });
}
