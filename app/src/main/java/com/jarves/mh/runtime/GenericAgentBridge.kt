package com.jarves.mh.runtime

import android.content.Context
import android.util.Log
import androidx.core.content.ContextCompat
import com.jarves.mh.model.ChatMessage
import com.jarves.mh.model.ChangeItem
import com.jarves.mh.model.DiffLine
import com.jarves.mh.model.DiffLineType
import com.jarves.mh.model.ProjectKind
import com.jarves.mh.model.ProviderProfile
import com.jarves.mh.model.ProviderProtocol
import com.jarves.mh.model.RuntimeEvent
import com.jarves.mh.model.ToolRequest
import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.json.JSONArray
import org.json.JSONObject

/**
 * Maya generic engine: a lightweight OpenAI-chat / Anthropic-messages agent loop
 * that runs tools through the same PRoot [RuntimeInstaller.process] bridge as the
 * terminal. Implements [RuntimeBridge] so the chat UI needs no changes.
 *
 * Protocol routing (mirrors ProviderApiClient):
 * - OPENROUTER, OPENAI_CHAT, OPENAI_RESPONSES -> OpenAI /chat/completions
 * - ANTHROPIC, ANTHROPIC_GATEWAY (DeepSeek/Kimi/Custom), CLAUDE_LOGIN -> Anthropic /v1/messages
 */
class GenericAgentBridge(
    private val context: Context,
    private val secretFor: (ProviderProfile) -> String?,
) : RuntimeBridge {
    private val installer = RuntimeInstaller(context)
    private val eventBus = MutableSharedFlow<RuntimeEvent>(extraBufferCapacity = 64)
    override val events: Flow<RuntimeEvent> = eventBus
    private val projectRoots = ConcurrentHashMap<String, String>()
    private val finishedSessions = ConcurrentHashMap.newKeySet<String>()

    @Volatile private var activeProcess: Process? = null
    @Volatile private var activeSessionId: String? = null
    @Volatile private var userStopRequested: Boolean = false
    @Volatile private var activeProjectSlug: String? = null
    @Volatile private var foregroundResultPosted: Boolean = false

    override suspend fun startSession(
        projectId: String,
        projectSlug: String,
        projectKind: ProjectKind,
        prompt: String,
        conversationHistory: List<ChatMessage>,
        provider: ProviderProfile,
    ): String = withContext(Dispatchers.IO + NonCancellable) {
        val sessionId = UUID.randomUUID().toString()
        finishedSessions.remove(sessionId)
        activeSessionId = sessionId
        userStopRequested = false
        activeProjectSlug = projectSlug
        foregroundResultPosted = false
        eventBus.emit(RuntimeEvent.SessionStarted(sessionId))
        val secret = secretFor(provider).orEmpty()
        if (secret.isBlank()) {
            eventBus.emit(RuntimeEvent.SessionFailed(sessionId, "No API key is saved for ${provider.kind.title}."))
            activeSessionId = null
            return@withContext sessionId
        }
        runCatching {
            RuntimeTaskController.stopAction = {
                userStopRequested = true
                val running = activeProcess
                if (running != null) {
                    Thread {
                        running.destroy()
                        Thread.sleep(500)
                        if (running.isAlive) running.destroyForcibly()
                    }.start()
                }
            }
            startForegroundRuntime(projectSlug)
            val installed = installer.installedRuntime()
            installer.ensureSettingsAndHooks()
            val workspace = ensureWorkspace(projectId)
            createCheckpoint(projectId, workspace)
            val before = snapshot(workspace)
            val guestWorkspacePath = "/workspace/$projectSlug"
            runAgentLoop(sessionId, projectId, workspace, guestWorkspacePath, projectKind, prompt, conversationHistory, provider, secret)
            val changed = changedFiles(workspace, before)
            if (changed.isNotEmpty()) {
                saveChangedPaths(projectId, changed)
                val details = loadPendingChanges(projectId)
                eventBus.emit(RuntimeEvent.FilesChanged(sessionId, details))
            } else if (!File(checkpointDir(projectId), "changes.json").isFile) {
                acceptLastChanges(projectId)
            }
            emitCompletedOnce(sessionId)
            finishForegroundRuntime(true, projectSlug, "Task finished in $projectSlug.")
        }.onFailure { error ->
            Log.e("GenericBridge", "Session failed", error)
            val message = error.message.orEmpty().take(500).ifBlank { "The generic engine could not finish the task." }
            if (userStopRequested) {
                emitFailureOnce(sessionId, "Stopped by user")
                cancelForegroundRuntime()
            } else {
                emitFailureOnce(sessionId, message)
                finishForegroundRuntime(false, projectSlug, message)
            }
        }
        activeProcess = null
        activeSessionId = null
        RuntimeTaskController.stopAction = null
        sessionId
    }

    override suspend fun respondToApproval(request: ToolRequest, approved: Boolean) = withContext(Dispatchers.IO) {
        // Auto-allow engine: nothing pending. Kept for interface compatibility.
        eventBus.emit(
            if (approved) RuntimeEvent.ToolApproved(request.sessionId, request.approvalId)
            else RuntimeEvent.ToolRejected(request.sessionId, request.approvalId),
        )
    }

    override suspend fun stopSession(sessionId: String) = withContext(Dispatchers.IO) {
        if (activeSessionId == sessionId) {
            userStopRequested = true
            activeProcess?.destroy()
            delay(500)
            if (activeProcess?.isAlive == true) activeProcess?.destroyForcibly()
            emitFailureOnce(sessionId, "Stopped by user")
        }
    }

    override suspend fun stopActiveSession() {
        activeSessionId?.let { stopSession(it) }
    }

    override suspend fun undoLastChanges(projectId: String): Boolean = withContext(Dispatchers.IO) {
        val checkpoint = checkpointDir(projectId)
        val backup = File(checkpoint, "project")
        val manifest = File(checkpoint, "changes.json")
        if (!backup.isDirectory || !manifest.isFile) return@withContext false
        val workspace = ensureWorkspace(projectId)
        val paths = runCatching {
            val array = JSONArray(manifest.readText())
            (0 until array.length()).map(array::getString)
        }.getOrElse { return@withContext false }.filterNot(::isInternalRuntimePath)
        paths.forEach { path ->
            val target = safeWorkspaceFile(workspace, path)
            val original = safeWorkspaceFile(backup, path)
            if (original.isFile) {
                target.parentFile?.mkdirs()
                original.copyTo(target, overwrite = true)
            } else if (target.isFile) {
                target.delete()
            }
        }
        checkpoint.deleteRecursively()
        true
    }

    override suspend fun acceptLastChanges(projectId: String) {
        withContext(Dispatchers.IO) {
            checkpointDir(projectId).deleteRecursively()
        }
    }

    override suspend fun loadPendingChanges(projectId: String): List<ChangeItem> = withContext(Dispatchers.IO) {
        val workspace = ensureWorkspace(projectId)
        val paths = readChangedPaths(projectId).filterNot(::isInternalRuntimePath)
        if (paths.isEmpty()) emptyList() else buildChangeDetails(projectId, workspace, paths)
    }

    override suspend fun undoFileChange(projectId: String, path: String): Boolean = withContext(Dispatchers.IO) {
        if (isInternalRuntimePath(path) || path !in readChangedPaths(projectId)) return@withContext false
        val workspace = ensureWorkspace(projectId)
        val backup = File(checkpointDir(projectId), "project")
        val target = safeWorkspaceFile(workspace, path)
        val original = safeWorkspaceFile(backup, path)
        if (original.isFile) {
            target.parentFile?.mkdirs()
            original.copyTo(target, overwrite = true)
        } else if (target.isFile) {
            target.delete()
        }
        removeChangedPath(projectId, path)
        true
    }

    override suspend fun acceptFileChange(projectId: String, path: String): Boolean = withContext(Dispatchers.IO) {
        if (isInternalRuntimePath(path) || path !in readChangedPaths(projectId)) return@withContext false
        val workspace = ensureWorkspace(projectId)
        val backup = File(checkpointDir(projectId), "project")
        val current = safeWorkspaceFile(workspace, path)
        val baseline = safeWorkspaceFile(backup, path)
        if (current.isFile) {
            baseline.parentFile?.mkdirs()
            current.copyTo(baseline, overwrite = true)
        } else if (baseline.isFile) {
            baseline.delete()
        }
        removeChangedPath(projectId, path)
        true
    }

    fun configureProjectRoot(projectId: String, rootPath: String) {
        val normalized = rootPath.trim().trim('/')
        require(normalized.isBlank() || (!normalized.contains("..") && !normalized.startsWith('/'))) {
            "Unsafe project root"
        }
        val previous = projectRoots.put(projectId, normalized).orEmpty()
        if (previous != normalized) checkpointDir(projectId).deleteRecursively()
    }

    // ---------- agent loop ----------

    private suspend fun runAgentLoop(
        sessionId: String,
        projectId: String,
        workspace: File,
        guestWorkspacePath: String,
        projectKind: ProjectKind,
        prompt: String,
        history: List<ChatMessage>,
        provider: ProviderProfile,
        secret: String,
    ) {
        val useAnthropic = provider.kind.protocol == ProviderProtocol.ANTHROPIC ||
            provider.kind.protocol == ProviderProtocol.ANTHROPIC_GATEWAY ||
            provider.kind.protocol == ProviderProtocol.CLAUDE_LOGIN
        val progressLog = readProgressLog(workspace)
        if (useAnthropic) {
            runAnthropicLoop(sessionId, workspace, guestWorkspacePath, projectKind, prompt, history, provider, secret, progressLog)
        } else {
            runOpenAiLoop(sessionId, workspace, guestWorkspacePath, projectKind, prompt, history, provider, secret, progressLog)
        }
    }

    private fun systemPrompt(guestWorkspacePath: String, projectKind: ProjectKind, progressLog: String?): String = buildString {
        appendLine("You are a coding agent running on an Android phone inside an Ubuntu 20.04 PRoot sandbox.")
        if (projectKind == ProjectKind.QUICK_PROJECT) {
            appendLine("This is a lightweight project workspace at $guestWorkspacePath.")
            appendLine("Respond conversationally, and use tools whenever they are useful. Keep every file and command inside this project workspace.")
        } else {
            appendLine("The current working directory $guestWorkspacePath is the project root.")
            appendLine("Create and edit project files directly in this directory. Do not create another outer project folder unless the user explicitly asks for one.")
        }
        appendLine("You have tools: Bash (run shell commands in the project root), Write (create files), Edit (modify files), Read (read files), Glob (find files by pattern), Grep (search file contents).")
        appendLine("Prefer small verifiable steps. After writing code, run it or check it with Bash when possible.")
        appendLine("If this is an Android project, the phone already provides JDK 17, Android SDK 36, ARM64 Build Tools 35.0.0, Gradle 8.14.3, and an offline Maven repository. Use the installed `gradle` command.")
        appendLine("Never invent tool results. If a tool returns an error, report it honestly and try a different approach.")
        if (!progressLog.isNullOrBlank()) {
            appendLine("<session_progress>")
            appendLine("Steps already completed in earlier turns or sessions. Do NOT repeat them — verify current file state and continue from here:")
            appendLine(progressLog)
            appendLine("</session_progress>")
        }
        appendLine("At the start of a turn, check the current file state (Glob or ls via Bash) before acting, then continue — do not redo completed steps.")
    }

    private fun historyMessages(history: List<ChatMessage>, currentPrompt: String): List<JSONObject> {
        val out = mutableListOf<JSONObject>()
        val prior = history
            .filter { msg ->
                (msg.fromUser || !msg.text.startsWith("Hi! Tell me")) &&
                    !msg.text.startsWith("Failed to") &&
                    !msg.text.startsWith("Error:") &&
                    !msg.text.contains("API Error")
            }
            .dropLast(1)
            .takeLast(10)
        for (msg in prior) {
            out += JSONObject()
                .put("role", if (msg.fromUser) "user" else "assistant")
                .put("content", msg.text.take(4000))
        }
        out += JSONObject().put("role", "user").put("content", currentPrompt)
        return out
    }

    // ----- OpenAI-chat branch -----

    private fun openAiEndpoint(provider: ProviderProfile): String {
        val base = provider.baseUrl.trim().trimEnd('/')
        return when (provider.kind.protocol) {
            ProviderProtocol.OPENROUTER -> "$base/v1/chat/completions"
            ProviderProtocol.OPENAI_RESPONSES -> if (base.endsWith("/chat/completions")) base else "$base/chat/completions"
            else -> if (base.endsWith("/chat/completions")) base else "$base/chat/completions"
        }
    }

    private fun openAiTools(): JSONArray = JSONArray().apply {
        put(toolDef("Bash", "Run a shell command in the project root. Returns combined stdout/stderr and exit code.",
            JSONObject().put("type", "object").put("properties", JSONObject()
                .put("command", JSONObject().put("type", "string").put("description", "Shell command to run"))
            ).put("required", JSONArray().put("command"))))
        put(toolDef("Write", "Create or overwrite a file with the given content. Parent directories are created.",
            JSONObject().put("type", "object").put("properties", JSONObject()
                .put("file_path", JSONObject().put("type", "string"))
                .put("content", JSONObject().put("type", "string"))
            ).put("required", JSONArray().put("file_path").put("content"))))
        put(toolDef("Edit", "Replace old_string with new_string in a file. old_string must match exactly once.",
            JSONObject().put("type", "object").put("properties", JSONObject()
                .put("file_path", JSONObject().put("type", "string"))
                .put("old_string", JSONObject().put("type", "string"))
                .put("new_string", JSONObject().put("type", "string"))
            ).put("required", JSONArray().put("file_path").put("old_string").put("new_string"))))
        put(toolDef("Read", "Read a file, optionally with offset (1-based) and limit of lines.",
            JSONObject().put("type", "object").put("properties", JSONObject()
                .put("file_path", JSONObject().put("type", "string"))
                .put("offset", JSONObject().put("type", "integer"))
                .put("limit", JSONObject().put("type", "integer"))
            ).put("required", JSONArray().put("file_path"))))
        put(toolDef("Glob", "Find files matching a glob pattern like *.py or **/*.kt, relative to project root.",
            JSONObject().put("type", "object").put("properties", JSONObject()
                .put("pattern", JSONObject().put("type", "string"))
            ).put("required", JSONArray().put("pattern"))))
        put(toolDef("Grep", "Search file contents for a regex pattern. Returns file:line matches.",
            JSONObject().put("type", "object").put("properties", JSONObject()
                .put("pattern", JSONObject().put("type", "string"))
                .put("path", JSONObject().put("type", "string").put("description", "Optional subdirectory"))
                .put("include", JSONObject().put("type", "string").put("description", "Optional glob filter like *.py"))
            ).put("required", JSONArray().put("pattern"))))
    }

    private fun toolDef(name: String, description: String, parameters: JSONObject): JSONObject =
        JSONObject().put("type", "function")
            .put("function", JSONObject().put("name", name).put("description", description).put("parameters", parameters))

    private suspend fun runOpenAiLoop(
        sessionId: String,
        workspace: File,
        guestWorkspacePath: String,
        projectKind: ProjectKind,
        prompt: String,
        history: List<ChatMessage>,
        provider: ProviderProfile,
        secret: String,
        progressLog: String?,
    ) {
        val endpoint = openAiEndpoint(provider)
        val messages = mutableListOf<JSONObject>()
        messages += JSONObject().put("role", "system").put("content", systemPrompt(guestWorkspacePath, projectKind, progressLog))
        messages += historyMessages(history, prompt)
        val tools = openAiTools()
        repeat(MAX_TURNS) { turn ->
            if (userStopRequested) throw IllegalStateException("Stopped by user")
            val body = JSONObject()
                .put("model", provider.model.trim())
                .put("messages", JSONArray(messages))
                .put("tools", tools)
                .put("tool_choice", "auto")
            val response = postJson(endpoint, body, secret, provider, openAiHeaders = true)
            val choice = response.optJSONArray("choices")?.optJSONObject(0)
                ?: throw IllegalStateException(providerError(response.toString()))
            val message = choice.optJSONObject("message") ?: JSONObject()
            val text = message.optString("content")
            if (text.isNotBlank()) {
                eventBus.emit(RuntimeEvent.AssistantDelta(sessionId, sanitizeForDisplay(text).take(4000)))
            }
            val toolCalls = message.optJSONArray("tool_calls")
            if (toolCalls == null || toolCalls.length() == 0) {
                if (text.isBlank()) eventBus.emit(RuntimeEvent.AssistantDelta(sessionId, "Done."))
                return
            }
            messages += JSONObject().put("role", "assistant").put("content", text.ifBlank { JSONObject.NULL })
                .put("tool_calls", toolCalls)
            for (i in 0 until toolCalls.length()) {
                val call = toolCalls.getJSONObject(i)
                val callId = call.optString("id").ifBlank { "call_$turn$i" }
                val function = call.optJSONObject("function") ?: JSONObject()
                val name = function.optString("name")
                val args = runCatching { JSONObject(function.optString("arguments", "{}")) }.getOrDefault(JSONObject())
                eventBus.emit(RuntimeEvent.ToolStarted(sessionId, name.ifBlank { "Tool" }, sanitizeForDisplay(toolPreview(name, args))))
                val result = executeTool(name, args, workspace, guestWorkspacePath, sessionId)
                eventBus.emit(RuntimeEvent.ToolCompleted(sessionId, name.ifBlank { "Tool" }, sanitizeForDisplay(result.take(600))))
                messages += JSONObject().put("role", "tool").put("tool_call_id", callId)
                    .put("content", result.take(MAX_TOOL_OUTPUT))
            }
        }
        throw IllegalStateException("Reached the $MAX_TURNS-step limit. The task is too large for one run; ask to continue.")
    }

    // ----- Anthropic branch -----

    private fun anthropicEndpoint(provider: ProviderProfile): String {
        val base = provider.baseUrl.trim().trimEnd('/')
        return if (base.endsWith("/v1/messages")) base else "$base/v1/messages"
    }

    private fun anthropicTools(): JSONArray = JSONArray().apply {
        fun schema(props: JSONObject, required: List<String>): JSONObject =
            JSONObject().put("type", "object").put("properties", props)
                .put("required", JSONArray(required))
        put(JSONObject().put("name", "Bash")
            .put("description", "Run a shell command in the project root. Returns combined stdout/stderr and exit code.")
            .put("input_schema", schema(JSONObject().put("command", JSONObject().put("type", "string")), listOf("command"))))
        put(JSONObject().put("name", "Write")
            .put("description", "Create or overwrite a file with the given content. Parent directories are created.")
            .put("input_schema", schema(JSONObject()
                .put("file_path", JSONObject().put("type", "string"))
                .put("content", JSONObject().put("type", "string")), listOf("file_path", "content"))))
        put(JSONObject().put("name", "Edit")
            .put("description", "Replace old_string with new_string in a file. old_string must match exactly once.")
            .put("input_schema", schema(JSONObject()
                .put("file_path", JSONObject().put("type", "string"))
                .put("old_string", JSONObject().put("type", "string"))
                .put("new_string", JSONObject().put("type", "string")), listOf("file_path", "old_string", "new_string"))))
        put(JSONObject().put("name", "Read")
            .put("description", "Read a file, optionally with offset (1-based) and limit of lines.")
            .put("input_schema", schema(JSONObject()
                .put("file_path", JSONObject().put("type", "string"))
                .put("offset", JSONObject().put("type", "integer"))
                .put("limit", JSONObject().put("type", "integer")), listOf("file_path"))))
        put(JSONObject().put("name", "Glob")
            .put("description", "Find files matching a glob pattern like *.py, relative to project root.")
            .put("input_schema", schema(JSONObject().put("pattern", JSONObject().put("type", "string")), listOf("pattern"))))
        put(JSONObject().put("name", "Grep")
            .put("description", "Search file contents for a regex pattern. Returns file:line matches.")
            .put("input_schema", schema(JSONObject()
                .put("pattern", JSONObject().put("type", "string"))
                .put("path", JSONObject().put("type", "string"))
                .put("include", JSONObject().put("type", "string")), listOf("pattern"))))
    }

    private suspend fun runAnthropicLoop(
        sessionId: String,
        workspace: File,
        guestWorkspacePath: String,
        projectKind: ProjectKind,
        prompt: String,
        history: List<ChatMessage>,
        provider: ProviderProfile,
        secret: String,
        progressLog: String?,
    ) {
        val endpoint = anthropicEndpoint(provider)
        val messages = mutableListOf<JSONObject>()
        for (m in historyMessages(history, prompt)) {
            val role = m.optString("role")
            messages += JSONObject().put("role", if (role == "assistant") "assistant" else "user")
                .put("content", m.optString("content"))
        }
        val tools = anthropicTools()
        repeat(MAX_TURNS) {
            if (userStopRequested) throw IllegalStateException("Stopped by user")
            val body = JSONObject()
                .put("model", provider.model.trim())
                .put("max_tokens", 4096)
                .put("system", systemPrompt(guestWorkspacePath, projectKind, progressLog))
                .put("messages", JSONArray(messages))
                .put("tools", tools)
            val response = postJson(endpoint, body, secret, provider, openAiHeaders = false)
            if (response.has("error")) {
                throw IllegalStateException(providerError(response.toString()))
            }
            val content = response.optJSONArray("content") ?: JSONArray()
            val toolUses = mutableListOf<JSONObject>()
            val textParts = mutableListOf<String>()
            for (i in 0 until content.length()) {
                val block = content.optJSONObject(i) ?: continue
                when (block.optString("type")) {
                    "text" -> block.optString("text").takeIf(String::isNotBlank)?.let(textParts::add)
                    "tool_use" -> toolUses += block
                }
            }
            val text = textParts.joinToString("\n")
            if (text.isNotBlank()) {
                eventBus.emit(RuntimeEvent.AssistantDelta(sessionId, sanitizeForDisplay(text).take(4000)))
            }
            if (toolUses.isEmpty()) {
                if (text.isBlank()) eventBus.emit(RuntimeEvent.AssistantDelta(sessionId, "Done."))
                return
            }
            messages += JSONObject().put("role", "assistant").put("content", content)
            val results = JSONArray()
            for (block in toolUses) {
                val id = block.optString("id")
                val name = block.optString("name")
                val input = block.optJSONObject("input") ?: JSONObject()
                eventBus.emit(RuntimeEvent.ToolStarted(sessionId, name.ifBlank { "Tool" }, sanitizeForDisplay(toolPreview(name, input))))
                val result = executeTool(name, input, workspace, guestWorkspacePath, sessionId)
                eventBus.emit(RuntimeEvent.ToolCompleted(sessionId, name.ifBlank { "Tool" }, sanitizeForDisplay(result.take(600))))
                results.put(JSONObject().put("type", "tool_result").put("tool_use_id", id).put("content", result.take(MAX_TOOL_OUTPUT)))
            }
            messages += JSONObject().put("role", "user").put("content", results)
        }
        throw IllegalStateException("Reached the $MAX_TURNS-step limit. The task is too large for one run; ask to continue.")
    }

    // ----- HTTP -----

    private fun postJson(endpoint: String, body: JSONObject, secret: String, provider: ProviderProfile, openAiHeaders: Boolean): JSONObject {
        val connection = URL(endpoint).openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "POST"
            connection.connectTimeout = 20_000
            connection.readTimeout = 180_000
            connection.doOutput = true
            connection.setRequestProperty("Content-Type", "application/json")
            connection.setRequestProperty("Authorization", "Bearer $secret")
            if (openAiHeaders) {
                if (provider.kind.protocol == ProviderProtocol.OPENROUTER) {
                    connection.setRequestProperty("HTTP-Referer", "https://github.com/techjarves/Mobile-Harness")
                    connection.setRequestProperty("X-Title", "Mobile Harness")
                }
            } else {
                connection.setRequestProperty("x-api-key", secret)
                connection.setRequestProperty("anthropic-version", "2023-06-01")
            }
            connection.outputStream.use { it.write(body.toString().toByteArray()) }
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val text = stream?.bufferedReader()?.use { it.readText() }.orEmpty()
            if (code !in 200..299) throw IllegalStateException(providerError(text).ifBlank { "Provider returned HTTP $code" })
            return JSONObject(text)
        } finally {
            connection.disconnect()
        }
    }

    private fun providerError(body: String): String = runCatching {
        val root = JSONObject(body)
        root.optJSONObject("error")?.optString("message").orEmpty().ifBlank {
            root.optString("message").ifBlank { root.optString("error").ifBlank { body.take(500) } }
        }
    }.getOrDefault(body.take(500))

    // ----- tools -----

    private fun toolPreview(name: String, args: JSONObject): String = when (name) {
        "Bash" -> args.optString("command").ifBlank { "Running Bash" }
        "Write", "Edit", "Read" -> args.optString("file_path").ifBlank { "Running $name" }
        "Glob", "Grep" -> args.optString("pattern").ifBlank { "Running $name" }
        else -> "Running $name"
    }

    private suspend fun executeTool(name: String, args: JSONObject, workspace: File, guestWorkspacePath: String, sessionId: String): String {
        val result = try {
            when (name) {
                "Bash" -> {
                    val command = args.optString("command")
                    if (command.isBlank()) "Error: missing command"
                    else runGuestBash(workspace, guestWorkspacePath, command)
                }
                "Write" -> {
                    val path = args.optString("file_path")
                    if (path.isBlank()) "Error: missing file_path"
                    else {
                        val target = safeWorkspaceFile(workspace, normalizeGuestPath(path, guestWorkspacePath))
                        target.parentFile?.mkdirs()
                        target.writeText(args.optString("content"))
                        "Wrote ${target.relativeTo(workspace).invariantSeparatorsPath} (${target.length()} bytes)"
                    }
                }
                "Edit" -> {
                    val path = args.optString("file_path")
                    val old = args.optString("old_string")
                    val new = args.optString("new_string")
                    if (path.isBlank() || old.isEmpty()) "Error: file_path and old_string are required"
                    else {
                        val target = safeWorkspaceFile(workspace, normalizeGuestPath(path, guestWorkspacePath))
                        if (!target.isFile) "Error: file not found: $path"
                        else {
                            val current = target.readText()
                            val occurrences = current.split(old, limit = Int.MAX_VALUE).size - 1
                            if (occurrences == 0) "Error: old_string not found in $path"
                            else if (occurrences > 1) "Error: old_string matches $occurrences times in $path; make it unique"
                            else {
                                target.writeText(current.replace(old, new))
                                "Edited $path"
                            }
                        }
                    }
                }
                "Read" -> {
                    val path = args.optString("file_path")
                    if (path.isBlank()) "Error: missing file_path"
                    else {
                        val target = safeWorkspaceFile(workspace, normalizeGuestPath(path, guestWorkspacePath))
                        if (!target.isFile) "Error: file not found: $path"
                        else {
                            val lines = target.readText().split('\n')
                            val offset = (args.optInt("offset", 1)).coerceAtLeast(1)
                            val limit = (args.optInt("limit", 200)).coerceIn(1, 500)
                            lines.drop(offset - 1).take(limit).joinToString("\n").take(MAX_TOOL_OUTPUT)
                        }
                    }
                }
                "Glob" -> {
                    val pattern = args.optString("pattern").ifBlank { "*" }
                    globFiles(workspace, pattern).take(100).joinToString("\n").ifBlank { "No files match $pattern" }
                }
                "Grep" -> {
                    val pattern = args.optString("pattern")
                    if (pattern.isBlank()) "Error: missing pattern"
                    else grepFiles(workspace, pattern, args.optString("path"), args.optString("include")).take(50).joinToString("\n").ifBlank { "No matches for $pattern" }
                }
                else -> "Error: unknown tool $name"
            }
        } catch (error: Exception) {
            if (error.message?.contains("Unsafe") == true) "Error: unsafe path rejected"
            else "Error: ${error.message?.take(300) ?: "tool failed"}"
        }
        appendProgressLog(workspace, name, args, result)
        return result
    }

    private fun normalizeGuestPath(path: String, guestWorkspacePath: String): String {
        var clean = path.trim().replace('\\', '/')
        if (clean.startsWith(guestWorkspacePath)) clean = clean.removePrefix(guestWorkspacePath).trimStart('/')
        clean = clean.trimStart('/').trimStart('.').trimStart('/')
        if (clean.isBlank() || clean.contains("..")) throw IllegalArgumentException("Unsafe workspace path")
        return clean
    }

    private suspend fun runGuestBash(workspace: File, guestWorkspacePath: String, command: String): String {
        val installed = installer.installedRuntime()
        val process = installer.process(
            installed.proot,
            installed.rootfs,
            workspace,
            emptyMap(),
            listOf("/usr/bin/bash", "-lc", command),
            guestWorkspacePath = guestWorkspacePath,
        )
        activeProcess = process
        if (userStopRequested) process.destroy()
        val native = process as? NativeSpawnProcess
        try {
            withTimeout(TOOL_TIMEOUT_MS) {
                while (process.isAlive) delay(100)
            }
        } catch (timeout: Exception) {
            process.destroy()
            delay(500)
            if (process.isAlive) process.destroyForcibly()
            val partial = readProcessOutput(native).take(MAX_TOOL_OUTPUT)
            return (if (partial.isNotBlank()) "$partial\n" else "") + "[timed out after ${TOOL_TIMEOUT_MS / 1000}s]"
        } finally {
            if (process.isAlive) process.destroy()
        }
        val exit = process.waitFor()
        val output = readProcessOutput(native).take(MAX_TOOL_OUTPUT)
        val trimmed = output.trimEnd()
        return if (trimmed.isEmpty()) "[exit $exit]" else "$trimmed\n[exit $exit]"
    }

    private fun readProcessOutput(native: NativeSpawnProcess?): String = runCatching {
        val file = native?.outputFile ?: return ""
        val decoder = Charsets.UTF_8.newDecoder()
            .onMalformedInput(java.nio.charset.CodingErrorAction.REPLACE)
            .onUnmappableCharacter(java.nio.charset.CodingErrorAction.REPLACE)
        decoder.decode(java.nio.ByteBuffer.wrap(file.readBytes())).toString()
    }.getOrDefault("")

    private fun globFiles(workspace: File, pattern: String): List<String> {
        val regex = globToRegex(pattern.trim().trimStart('/'))
        return workspace.walkTopDown()
            .filter { it.isFile && !isInternalRuntimePath(it.relativeTo(workspace).invariantSeparatorsPath) }
            .map { it.relativeTo(workspace).invariantSeparatorsPath }
            .filter { regex.containsMatchIn(it) || regex.matches(it) }
            .sorted()
            .toList()
    }

    private fun globToRegex(glob: String): Regex {
        val sb = StringBuilder()
        var i = 0
        while (i < glob.length) {
            when (val c = glob[i]) {
                '*' -> if (i + 1 < glob.length && glob[i + 1] == '*') {
                    sb.append(".*")
                    i += if (i + 2 < glob.length && glob[i + 2] == '/') 3 else 2
                } else {
                    sb.append("[^/]*")
                    i++
                }
                '?' -> {
                    sb.append("[^/]")
                    i++
                }
                '.', '(', ')', '+', '|', '^', '$', '{', '}', '[', ']', '\\' -> {
                    sb.append('\\').append(c)
                    i++
                }
                else -> {
                    sb.append(c)
                    i++
                }
            }
        }
        return Regex(sb.toString())
    }

    private fun grepFiles(workspace: File, pattern: String, subPath: String, include: String): List<String> {
        val regex = runCatching { Regex(pattern) }.getOrElse { Regex(Regex.escape(pattern)) }
        val includeRegex = include.takeIf(String::isNotBlank)?.let { globToRegex(it.trim()) }
        val root = if (subPath.isNotBlank()) {
            runCatching { safeWorkspaceFile(workspace, normalizeGuestPath(subPath, "/workspace")) }.getOrNull() ?: workspace
        } else workspace
        if (!root.exists()) return emptyList()
        val out = mutableListOf<String>()
        root.walkTopDown()
            .filter { it.isFile && it.length() < 200_000 && !isInternalRuntimePath(it.relativeTo(workspace).invariantSeparatorsPath) }
            .forEach { file ->
                val rel = file.relativeTo(workspace).invariantSeparatorsPath
                if (includeRegex != null && !includeRegex.containsMatchIn(rel.substringAfterLast('/'))) return@forEach
                runCatching {
                    file.readLines().forEachIndexed { index, line ->
                        if (regex.containsMatchIn(line)) {
                            out += "$rel:${index + 1}: ${line.trim().take(200)}"
                            if (out.size >= 200) return out
                        }
                    }
                }
            }
        return out
    }

    // ----- checkpoints / diffs (mirrors ClaudeRuntimeBridge) -----

    private fun ensureWorkspace(projectId: String): File {
        val base = File(context.filesDir, "workspaces/$projectId").apply { mkdirs() }.canonicalFile
        val rootPath = projectRoots[projectId].orEmpty()
        if (rootPath.isBlank()) return base
        val selected = File(base, rootPath).canonicalFile
        require(selected.toPath().startsWith(base.toPath())) { "Unsafe project root" }
        return selected.apply { mkdirs() }
    }

    private fun checkpointDir(projectId: String) = File(context.filesDir, "checkpoints/$projectId/latest")

    private fun createCheckpoint(projectId: String, workspace: File) {
        val checkpoint = checkpointDir(projectId)
        if (File(checkpoint, "project").isDirectory && File(checkpoint, "changes.json").isFile) return
        checkpoint.deleteRecursively()
        val backup = File(checkpoint, "project").apply { mkdirs() }
        val workspacePath = workspace.canonicalFile.toPath()
        workspace.walkTopDown()
            .onEnter { directory ->
                directory == workspace || (
                    !java.nio.file.Files.isSymbolicLink(directory.toPath()) &&
                        runCatching { directory.canonicalFile.toPath().startsWith(workspacePath) }.getOrDefault(false)
                    )
            }
            .filter {
                it.isFile &&
                    !isInternalRuntimePath(it.relativeTo(workspace).invariantSeparatorsPath) &&
                    !java.nio.file.Files.isSymbolicLink(it.toPath())
            }
            .forEach { source ->
                val relative = source.relativeTo(workspace).invariantSeparatorsPath
                val destination = safeWorkspaceFile(backup, relative)
                destination.parentFile?.mkdirs()
                source.copyTo(destination, overwrite = true)
            }
    }

    private fun saveChangedPaths(projectId: String, paths: List<String>) {
        val manifest = File(checkpointDir(projectId), "changes.json")
        manifest.parentFile?.mkdirs()
        val merged = (readChangedPaths(projectId) + paths)
            .filterNot(::isInternalRuntimePath)
            .distinct()
            .sorted()
        manifest.writeText(JSONArray(merged).toString())
    }

    private fun readChangedPaths(projectId: String): List<String> {
        val manifest = File(checkpointDir(projectId), "changes.json")
        if (!manifest.isFile) return emptyList()
        return runCatching {
            val array = JSONArray(manifest.readText())
            (0 until array.length()).map(array::getString)
        }.getOrDefault(emptyList())
    }

    private fun removeChangedPath(projectId: String, path: String) {
        val remaining = readChangedPaths(projectId).filterNot { it == path }
        if (remaining.isEmpty()) {
            checkpointDir(projectId).deleteRecursively()
        } else {
            File(checkpointDir(projectId), "changes.json").writeText(JSONArray(remaining).toString())
        }
    }

    private fun buildChangeDetails(projectId: String, workspace: File, paths: List<String>): List<ChangeItem> {
        val backup = File(checkpointDir(projectId), "project")
        return paths.map { path ->
            val before = safeWorkspaceFile(backup, path).takeIf(File::isFile)?.readBytes() ?: ByteArray(0)
            val after = safeWorkspaceFile(workspace, path).takeIf(File::isFile)?.readBytes() ?: ByteArray(0)
            val binary = before.any { it == 0.toByte() } || after.any { it == 0.toByte() }
            val (additions, deletions) = lineChanges(before, after)
            ChangeItem(path = path, additions = additions, deletions = deletions, diffLines = buildDiffLines(before, after), binary = binary)
        }
    }

    private fun buildDiffLines(beforeBytes: ByteArray, afterBytes: ByteArray): List<DiffLine> {
        if (beforeBytes.any { it == 0.toByte() } || afterBytes.any { it == 0.toByte() }) {
            return listOf(DiffLine(DiffLineType.INFO, "Binary file changed"))
        }
        val before = textLines(beforeBytes)
        val after = textLines(afterBytes)
        if (before.size > MAX_RENDERED_DIFF_LINES || after.size > MAX_RENDERED_DIFF_LINES) {
            return listOf(DiffLine(DiffLineType.INFO, "Diff is too large to display (${before.size} → ${after.size} lines). Undo and Keep still work."))
        }
        val lcs = Array(before.size + 1) { IntArray(after.size + 1) }
        for (oldIndex in before.lastIndex downTo 0) {
            for (newIndex in after.lastIndex downTo 0) {
                lcs[oldIndex][newIndex] = if (before[oldIndex] == after[newIndex]) {
                    lcs[oldIndex + 1][newIndex + 1] + 1
                } else {
                    maxOf(lcs[oldIndex + 1][newIndex], lcs[oldIndex][newIndex + 1])
                }
            }
        }
        val result = mutableListOf<DiffLine>()
        var oldIndex = 0
        var newIndex = 0
        while (oldIndex < before.size || newIndex < after.size) {
            when {
                oldIndex < before.size && newIndex < after.size && before[oldIndex] == after[newIndex] -> {
                    result += DiffLine(DiffLineType.CONTEXT, before[oldIndex], oldIndex + 1, newIndex + 1)
                    oldIndex++
                    newIndex++
                }
                newIndex < after.size && (oldIndex == before.size || lcs[oldIndex][newIndex + 1] >= lcs[oldIndex + 1][newIndex]) -> {
                    result += DiffLine(DiffLineType.ADDITION, after[newIndex], null, newIndex + 1)
                    newIndex++
                }
                oldIndex < before.size -> {
                    result += DiffLine(DiffLineType.DELETION, before[oldIndex], oldIndex + 1, null)
                    oldIndex++
                }
            }
        }
        return collapseUnchangedLines(result)
    }

    private fun collapseUnchangedLines(lines: List<DiffLine>): List<DiffLine> {
        val changedIndexes = lines.indices.filter { lines[it].type != DiffLineType.CONTEXT }
        if (changedIndexes.isEmpty()) return lines
        val visible = BooleanArray(lines.size)
        changedIndexes.forEach { changed ->
            for (index in maxOf(0, changed - DIFF_CONTEXT_LINES)..minOf(lines.lastIndex, changed + DIFF_CONTEXT_LINES)) {
                visible[index] = true
            }
        }
        val result = mutableListOf<DiffLine>()
        var index = 0
        while (index < lines.size) {
            if (visible[index]) {
                result += lines[index++]
            } else {
                val start = index
                while (index < lines.size && !visible[index]) index++
                result += DiffLine(DiffLineType.INFO, "… ${index - start} unchanged lines …")
            }
        }
        return result
    }

    private fun lineChanges(beforeBytes: ByteArray, afterBytes: ByteArray): Pair<Int, Int> {
        if (beforeBytes.any { it == 0.toByte() } || afterBytes.any { it == 0.toByte() }) {
            return (if (afterBytes.isNotEmpty()) 1 else 0) to (if (beforeBytes.isNotEmpty()) 1 else 0)
        }
        val before = textLines(beforeBytes)
        val after = textLines(afterBytes)
        if (before.size > MAX_DIFF_LINES || after.size > MAX_DIFF_LINES) {
            return maxOf(0, after.size - before.size) to maxOf(0, before.size - after.size)
        }
        var previous = IntArray(after.size + 1)
        before.forEach { oldLine ->
            val current = IntArray(after.size + 1)
            after.forEachIndexed { index, newLine ->
                current[index + 1] = if (oldLine == newLine) {
                    previous[index] + 1
                } else {
                    maxOf(previous[index + 1], current[index])
                }
            }
            previous = current
        }
        val common = previous[after.size]
        return (after.size - common) to (before.size - common)
    }

    private fun textLines(bytes: ByteArray): List<String> {
        if (bytes.isEmpty()) return emptyList()
        val lines = bytes.decodeToString().split('\n')
        return if (lines.lastOrNull().isNullOrEmpty()) lines.dropLast(1) else lines
    }

    private fun safeWorkspaceFile(root: File, relative: String): File {
        require(relative.isNotBlank() && !relative.startsWith('/')) { "Unsafe workspace path" }
        val file = File(root, relative)
        val rootPath = root.canonicalFile.toPath()
        val parentPath = (file.parentFile ?: root).canonicalFile.toPath()
        require(parentPath.startsWith(rootPath)) { "Workspace path escapes project" }
        return file
    }

    private fun snapshot(root: File): Map<String, String> = root.walkTopDown()
        .filter { it.isFile && !isInternalRuntimePath(it.relativeTo(root).invariantSeparatorsPath) }
        .associate { it.relativeTo(root).path to digest(it) }

    private fun changedFiles(root: File, before: Map<String, String>): List<String> {
        val after = snapshot(root)
        return (before.keys + after.keys).distinct().filter { before[it] != after[it] }.sorted()
    }

    private fun isInternalRuntimePath(path: String): Boolean {
        val normalized = path.replace('\\', '/')
        return normalized == ".claude" || normalized == ".claude.json" || normalized.startsWith(".claude/") ||
            normalized == ".mh" || normalized == PROGRESS_LOG_PATH || normalized.startsWith(".mh/")
    }

    // ----- session progress log (resume across stops/restarts) -----

    private fun readProgressLog(workspace: File): String? {
        return runCatching {
            val log = File(workspace, PROGRESS_LOG_PATH)
            if (!log.isFile) return null
            val lines = log.readLines().filter(String::isNotBlank).takeLast(PROGRESS_LOG_READ_LINES)
            if (lines.isEmpty()) null else lines.joinToString("\n")
        }.getOrNull()
    }

    private fun appendProgressLog(workspace: File, name: String, args: JSONObject, result: String) {
        runCatching {
            val summary = when (name) {
                "Bash" -> {
                    val cmd = args.optString("command").replace(Regex("\\s+"), " ").trim().take(220)
                    val tail = result.replace(Regex("\\s+"), " ").trim().take(220)
                    "Bash: `$cmd` → $tail"
                }
                "Write", "Edit" -> {
                    val path = args.optString("file_path").take(180)
                    val ok = !result.startsWith("Error:")
                    "$name: `$path` → ${if (ok) "ok" else result.take(220)}"
                }
                "Read", "Glob", "Grep" -> {
                    val what = args.optString("file_path").ifBlank {
                        args.optString("pattern").ifBlank { args.optString("path") }
                    }.take(180)
                    "$name: `$what` → ${result.replace(Regex("\\s+"), " ").trim().take(160)}"
                }
                else -> return
            }
            val log = File(workspace, PROGRESS_LOG_PATH)
            log.parentFile?.mkdirs()
            val existing = if (log.isFile) log.readLines() else emptyList()
            val trimmed = (existing + "- $summary").takeLast(PROGRESS_LOG_MAX_LINES)
            log.writeText(trimmed.joinToString("\n") + "\n")
        }
    }

    private fun digest(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun sanitizeForDisplay(value: String): String {
        return value
            .replace(Regex("sk-[A-Za-z0-9_-]{8,}"), "sk-••••")
            .replace(Regex("(?i)(authorization|api[_-]?key)\\s*[:=]\\s*[^\\s,}]+"), "$1: ••••")
            .replace(Regex("\\s+"), " ")
            .trim()
            .take(600)
    }

    private suspend fun emitCompletedOnce(sessionId: String) {
        if (finishedSessions.add(sessionId)) {
            eventBus.emit(RuntimeEvent.SessionCompleted(sessionId))
            finishForegroundRuntime(true, activeProjectSlug ?: "your project", "Task finished.")
        }
    }

    private suspend fun emitFailureOnce(sessionId: String, reason: String) {
        if (finishedSessions.add(sessionId)) {
            eventBus.emit(RuntimeEvent.SessionFailed(sessionId, reason))
            if (userStopRequested) {
                cancelForegroundRuntime()
            } else {
                finishForegroundRuntime(false, activeProjectSlug ?: "your project", reason)
            }
        }
    }

    private fun startForegroundRuntime(projectName: String) {
        ContextCompat.startForegroundService(
            context,
            android.content.Intent(context, RuntimeExecutionService::class.java)
                .setAction(RuntimeExecutionService.ACTION_START)
                .putExtra(RuntimeExecutionService.EXTRA_PROJECT_NAME, projectName),
        )
    }

    private fun finishForegroundRuntime(completed: Boolean, projectName: String, detail: String) {
        if (foregroundResultPosted) return
        foregroundResultPosted = true
        runCatching {
            context.startService(
                android.content.Intent(context, RuntimeExecutionService::class.java)
                    .setAction(if (completed) RuntimeExecutionService.ACTION_COMPLETE else RuntimeExecutionService.ACTION_FAILED)
                    .putExtra(RuntimeExecutionService.EXTRA_PROJECT_NAME, projectName)
                    .putExtra(RuntimeExecutionService.EXTRA_DETAIL, detail),
            )
        }.onFailure {
            context.stopService(android.content.Intent(context, RuntimeExecutionService::class.java))
        }
    }

    private fun cancelForegroundRuntime() {
        if (foregroundResultPosted) return
        foregroundResultPosted = true
        runCatching {
            context.startService(
                android.content.Intent(context, RuntimeExecutionService::class.java)
                    .setAction(RuntimeExecutionService.ACTION_CANCELLED),
            )
        }.onFailure {
            context.stopService(android.content.Intent(context, RuntimeExecutionService::class.java))
        }
    }

    companion object {
        private const val MAX_TURNS = 25
        private const val MAX_TOOL_OUTPUT = 12_000
        private const val TOOL_TIMEOUT_MS = 180_000L
        private const val MAX_DIFF_LINES = 2_000
        private const val MAX_RENDERED_DIFF_LINES = 600
        private const val DIFF_CONTEXT_LINES = 3
        private const val PROGRESS_LOG_PATH = ".mh/agent-log.md"
        private const val PROGRESS_LOG_MAX_LINES = 200
        private const val PROGRESS_LOG_READ_LINES = 40
    }
}
