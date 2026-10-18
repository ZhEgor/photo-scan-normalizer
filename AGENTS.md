# AGENTS.md — Autonomous Android Agent (OpenCode 2.0.30 + DeepSeek-V4.1-Flash)

## Rules
- Be brief

## Core Model & Capabilities
- **Primary Model:** `deepseek/deepseek-flash` (DeepSeek-V4.1-Flash)
- **Multimodal Advantage:** V4.1-Flash natively supports vision/images and a 1M context window. Therefore, the agent pipeline is configured to feed **both** the pruned structural accessibility tree JSON **and** the visual Set-of-Marks annotated screenshot to the model.
- **Verification Rule:** OpenCode 2.0 does not run Kotlin LSP diagnostics automatically. Always verify code changes by running:
  `./gradlew compileDebugKotlin`

---

## Architecture & Package Boundaries
- `core/Contracts.kt`: Shared immutable interfaces (`UiNode`, `ScreenState`, `AgentAction`, `ScreenController`, `PlannerEngine`). Do not modify contracts inside feature worktrees.
- `service/`: `AgentAccessibilityService`, `TreeFlattener` (parses nodes and captures screenshots via `takeScreenshot()`), and `GestureDispatcher`.
- `agent/`: `DeepSeekPlanner` (OkHttp client communicating with `https://api.deepseek.com/chat/completions` using `deepseek-flash` with native image/text payload and JSON mode) and `AgentLoopController` (500ms tree-hash debounce).
- `overlay/`: `OverlayService` (`TYPE_APPLICATION_OVERLAY` Jetpack Compose floating pill with prompt input, live V4.1-Flash `thought` display, emergency Stop button, and `AskConfirmation` dialog).

---

## OpenSpec Autonomous Execution Protocol
When tasked with features across your worktree (`feat-accessibility`, `feat-deepseek-brain`, or `feat-compose-overlay`):
1. Read the corresponding proposal in `openspec/changes/`.
2. Autonomously generate your `design.md` and granular `tasks.md` via OpenSpec.
3. Execute implementation tasks step-by-step with zero human intervention.
4. Run `./gradlew compileDebugKotlin` (and unit tests where applicable).
5. **Self-Healing Loop:** If a build or test fails, analyze the compiler traceback, rewrite the code to fix the error, and re-run compilation autonomously until everything passes successfully.
