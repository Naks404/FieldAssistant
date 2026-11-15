package com.example.myapplication

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.RggbChannelVector
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.video.Recorder
import androidx.camera.video.VideoCapture
import androidx.camera.view.PreviewView
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.snapping.rememberSnapFlingBehavior
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.example.myapplication.ui.theme.MyApplicationTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor
import org.json.JSONObject
import org.vosk.Recognizer
import org.vosk.android.StorageService
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executors
import kotlin.math.atan2
import kotlin.math.roundToInt
import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.ui.graphics.asImageBitmap
import kotlinx.coroutines.awaitCancellation

// =========================================================================
// TOP-LEVEL HELPER FUNCTIONS
// =========================================================================

fun assetFilePath(context: android.content.Context, assetName: String): String {
    val file = File(context.filesDir, assetName)
    if (file.exists() && file.length() > 0) return file.absolutePath
    context.assets.open(assetName).use { inputStream ->
        FileOutputStream(file).use { outputStream ->
            val buffer = ByteArray(4 * 1024)
            var read: Int
            while (inputStream.read(buffer).also { read = it } != -1) {
                outputStream.write(buffer, 0, read)
            }
            outputStream.flush()
        }
    }
    return file.absolutePath
}

fun argmax(array: FloatArray): Int {
    var maxIdx = 0
    var maxVal = array[0]
    for (i in 1 until array.size) {
        if (array[i] > maxVal) {
            maxVal = array[i]
            maxIdx = i
        }
    }
    return maxIdx
}
fun parseEffectPreset(context: android.content.Context, category: String, index: Int): String {
    try {
        val jsonString = context.assets.open("speech_effect_chain_v2.json").bufferedReader().use { it.readText() }
        val jsonObject = JSONObject(jsonString)

        val presetArray = jsonObject.getJSONArray(category)

        // Fix #2: Throw explicit error instead of silently clamping
        if (index < 0 || index >= presetArray.length()) {
            throw IndexOutOfBoundsException("Model predicted index $index for $category, but array size is ${presetArray.length()}")
        }

        val preset = presetArray.getJSONObject(index)
        if (preset.getBoolean("bypass")) return "Bypassed"

        val effects = preset.getJSONArray("effects")

        // Fix #8: Safe JSON parsing using optJSONObject
        return when (category) {
            "de_noise_presets" -> {
                val p = effects.optJSONObject(0)?.optJSONObject("params") ?: return "Applied (Params Missing)"
                "Noise Gate (Thresh: ${p.optDouble("threshold_db")}dB, Ratio: ${p.optDouble("ratio")}:1, Attack: ${p.optDouble("attack_ms")}ms, Release: ${p.optDouble("release_ms")}ms)"
            }
            "dynamic_presets" -> {
                val p = effects.optJSONObject(0)?.optJSONObject("params") ?: return "Applied (Params Missing)"
                var gainStr = ""
                if (effects.length() > 1) {
                    val g = effects.optJSONObject(1)?.optJSONObject("params")
                    if (g != null) gainStr = ", Makeup Gain: ${g.optDouble("gain_db")}dB"
                }
                "Compressor (Thresh: ${p.optDouble("threshold_db")}dB, Ratio: ${p.optDouble("ratio")}:1, Attack: ${p.optDouble("attack_ms")}ms, Release: ${p.optDouble("release_ms")}ms$gainStr)"
            }
            "eq_presets" -> {
                val low = effects.optJSONObject(0)?.optJSONObject("params") ?: return "Applied (Params Missing)"
                val high = effects.optJSONObject(1)?.optJSONObject("params")
                val mid = effects.optJSONObject(2)?.optJSONObject("params")
                "3-Band EQ\n" +
                        "  Low: ${low.optDouble("gain_db")}dB @ ${low.optDouble("cutoff_frequency_hz")}Hz (Q: ${low.optDouble("q")})\n" +
                        "  Mid: ${mid?.optDouble("gain_db")}dB @ ${mid?.optDouble("cutoff_frequency_hz")}Hz (Q: ${mid?.optDouble("q")})\n" +
                        "  High: ${high?.optDouble("gain_db")}dB @ ${high?.optDouble("cutoff_frequency_hz")}Hz (Q: ${high?.optDouble("q")})"
            }
            "de_esser_presets" -> {
                val p = effects.optJSONObject(0)?.optJSONObject("params") ?: return "Applied (Params Missing)"
                "De-Esser (Cutoff: ${p.optDouble("cutoff_frequency_hz")}Hz, Gain: ${p.optDouble("gain_db")}dB, Q: ${p.optDouble("q")})"
            }
            "reverb_presets" -> {
                val p = effects.optJSONObject(0)?.optJSONObject("params") ?: return "Applied (Params Missing)"
                "Reverb (Room: ${p.optDouble("room_size")}, Damp: ${p.optDouble("damping")}, Wet: ${p.optDouble("wet_level")}, Dry: ${p.optDouble("dry_level")}, Width: ${p.optDouble("width")}, Freeze: ${p.optDouble("freeze_mode")})"
            }
            "limiter_presets" -> {
                val p = effects.optJSONObject(0)?.optJSONObject("params") ?: return "Applied (Params Missing)"
                "Limiter (Thresh: ${p.optDouble("threshold_db")}dB, Release: ${p.optDouble("release_ms")}ms)"
            }
            else -> "Applied"
        }
    } catch (e: Exception) {
        Log.e("JSON_PARSER", "Failed to parse $category", e)
        throw e // Propagate to surface the ML error
    }
}

// =========================================================================
// APP NAVIGATION & CORE SCREENS
// =========================================================================

sealed class Screen {
    object Home : Screen()
    object Video : Screen()
    object Audio : Screen()
    object Gallery : Screen()
    data class Transcription(val filePath: String) : Screen()
    data class ImageViewer(val filePath: String) : Screen() // New route for JPGs
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MyApplicationTheme {
                AppNavigation()
            }
        }
    }
}

@Composable
fun AppNavigation() {
    var currentScreen by remember { mutableStateOf<Screen>(Screen.Home) }

    AnimatedContent(
        targetState = currentScreen,
        transitionSpec = { fadeIn(tween(300)) togetherWith fadeOut(tween(300)) },
        label = "ScreenTransition"
    ) { screen ->
        when (screen) {
            is Screen.Home -> HomeScreen(
                onNavigateToVideo = { currentScreen = Screen.Video },
                onNavigateToAudio = { currentScreen = Screen.Audio },
                onNavigateToGallery = { currentScreen = Screen.Gallery }
            )
            is Screen.Video -> VideoModeScreen(onNavigateBack = { currentScreen = Screen.Home })
            is Screen.Audio -> AudioModeScreen(
                onNavigateBack = { currentScreen = Screen.Home }
            )
            is Screen.Gallery -> GalleryScreen(
                onNavigateBack = { currentScreen = Screen.Home },
                onNavigateToTranscription = { filePath -> currentScreen = Screen.Transcription(filePath) },
                onNavigateToImageViewer = { filePath -> currentScreen = Screen.ImageViewer(filePath) }
            )
            is Screen.Transcription -> TranscriptionScreen(
                filePath = screen.filePath,
                onNavigateBack = { currentScreen = Screen.Gallery }
            )
            is Screen.ImageViewer -> ImageViewerScreen(
                filePath = screen.filePath,
                onNavigateBack = { currentScreen = Screen.Gallery }
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ImageViewerScreen(filePath: String, onNavigateBack: () -> Unit) {
    val bitmap = remember(filePath) { BitmapFactory.decodeFile(filePath)?.asImageBitmap() }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Image Viewer") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { paddingValues ->
        Box(
            modifier = Modifier.fillMaxSize().padding(paddingValues).background(Color.Black),
            contentAlignment = Alignment.Center
        ) {
            if (bitmap != null) {
                Image(bitmap = bitmap, contentDescription = "Captured Image", modifier = Modifier.fillMaxSize())
            } else {
                Text("Failed to load image.", color = Color.White)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(onNavigateToVideo: () -> Unit, onNavigateToAudio: () -> Unit, onNavigateToGallery: () -> Unit) {
    Scaffold(
        topBar = { TopAppBar(title = { Text("Field Assistant", fontWeight = FontWeight.SemiBold) }) }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 24.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(Icons.Default.Camera, contentDescription = "App Logo", modifier = Modifier.size(80.dp).padding(bottom = 16.dp), tint = MaterialTheme.colorScheme.primary)
            Text("Select Mode", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            Text("Real-time visual and acoustic processing.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 8.dp, bottom = 48.dp))
            ModeCard("Visual Capture", "Live framing & exposure feedback", Icons.Default.Videocam, onNavigateToVideo)
            Spacer(modifier = Modifier.height(16.dp))
            ModeCard("Acoustic Record", "Smart recording & auto-transcription", Icons.Default.Mic, onNavigateToAudio)
            Spacer(modifier = Modifier.height(16.dp))
            ModeCard("Gallery", "View and analyze recordings", Icons.Default.PhotoLibrary, onNavigateToGallery)
        }
    }
}

@Composable
fun ModeCard(title: String, description: String, icon: androidx.compose.ui.graphics.vector.ImageVector, onClick: () -> Unit) {
    ElevatedCard(
        modifier = Modifier.fillMaxWidth().clickable(onClick = onClick),
        elevation = CardDefaults.elevatedCardElevation(0.dp),
        colors = CardDefaults.elevatedCardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
    ) {
        Row(modifier = Modifier.fillMaxWidth().padding(20.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(modifier = Modifier.size(48.dp).clip(CircleShape).background(MaterialTheme.colorScheme.primaryContainer), contentAlignment = Alignment.Center) {
                Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.onPrimaryContainer)
            }
            Spacer(modifier = Modifier.width(16.dp))
            Column {
                Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                Text(description, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
@SuppressLint("MissingPermission")
@Composable
fun VideoModeScreen(onNavigateBack: () -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val configuration = LocalConfiguration.current
    val isLandscape = configuration.orientation == Configuration.ORIENTATION_LANDSCAPE

    var hasPermissions by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED &&
                    ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { perms ->
        hasPermissions = perms[Manifest.permission.CAMERA] == true && perms[Manifest.permission.RECORD_AUDIO] == true
    }
    LaunchedEffect(Unit) { if (!hasPermissions) permissionLauncher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)) }

    if (!hasPermissions) {
        PermissionDeniedScreen("Camera & Mic", onNavigateBack) { permissionLauncher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)) }
        return
    }

    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }
    var camera by remember { mutableStateOf<Camera?>(null) }
    var videoCapture by remember { mutableStateOf<VideoCapture<Recorder>?>(null) }
    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    var previewUseCase by remember { mutableStateOf<Preview?>(null) }
    var imageAnalysisUseCase by remember { mutableStateOf<ImageAnalysis?>(null) }
    var recording by remember { mutableStateOf<Recording?>(null) }
    var isVideoPaused by remember { mutableStateOf(false) }

    var exposureFeedback by remember { mutableStateOf<Map<String, String>>(emptyMap()) }
    var compositionFeedback by remember { mutableStateOf<Map<String, String>>(emptyMap()) }

    var isoValue by remember { mutableIntStateOf(400) }
    var focusDistance by remember { mutableFloatStateOf(0.0f) }
    var exposureTimeNs by remember { mutableLongStateOf(1_000_000_000L / 50) }
    var wbValue by remember { mutableIntStateOf(5000) }
    var activeSetting by remember { mutableStateOf("NONE") }

    val isoOptions = listOf(100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200).map { it to it.toString() }
    val shutterOptions = listOf<Pair<Long, String>>(
        30_000_000_000L to "30\"", 15_000_000_000L to "15\"", 8_000_000_000L to "8\"", 4_000_000_000L to "4\"", 2_000_000_000L to "2\"", 1_000_000_000L to "1\"",
        1000000000L/2 to "1/2", 1000000000L/4 to "1/4", 1000000000L/8 to "1/8", 1000000000L/15 to "1/15", 1000000000L/24 to "1/24", 1000000000L/25 to "1/25",
        1000000000L/30 to "1/30", 1000000000L/48 to "1/48", 1000000000L/50 to "1/50", 1000000000L/60 to "1/60", 1000000000L/96 to "1/96", 1000000000L/100 to "1/100",
        1000000000L/120 to "1/120", 1000000000L/125 to "1/125", 1000000000L/250 to "1/250", 1000000000L/500 to "1/500", 1000000000L/1000 to "1/1000"
    )
    val wbOptions = (20..80).map { it * 100 }.map { it to "${it}K" }
    val focusOptions = (0..20).map { i -> val v = i * 0.5f; v to if (i == 0) "Inf" else if (i == 20) "Macro" else "|" }

    val listStates = remember {
        mapOf(
            "WB" to LazyListState(firstVisibleItemIndex = wbOptions.indexOfFirst { it.first == wbValue }.coerceAtLeast(0)),
            "FOCUS" to LazyListState(firstVisibleItemIndex = focusOptions.indexOfFirst { it.first == focusDistance }.coerceAtLeast(0)),
            "ISO" to LazyListState(firstVisibleItemIndex = isoOptions.indexOfFirst { it.first == isoValue }.coerceAtLeast(0)),
            "SHUTTER" to LazyListState(firstVisibleItemIndex = shutterOptions.indexOfFirst { it.first == exposureTimeNs }.coerceAtLeast(0))
        )
    }

    val sensorManager = remember { context.getSystemService(android.content.Context.SENSOR_SERVICE) as SensorManager }
    val rotationAngle = remember { mutableFloatStateOf(0f) }
    DisposableEffect(Unit) {
        val sensor = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent?) { event?.let { rotationAngle.floatValue = Math.toDegrees(atan2(it.values[0].toDouble(), it.values[1].toDouble())).toFloat() } }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }
        sensor?.let { sensorManager.registerListener(listener, it, SensorManager.SENSOR_DELAY_UI) }
        onDispose { sensorManager.unregisterListener(listener) }
    }

    fun getCurrentDisplayRotation(): Int {
        val displayManager = context.getSystemService(android.content.Context.DISPLAY_SERVICE) as android.hardware.display.DisplayManager
        val display = displayManager.getDisplay(android.view.Display.DEFAULT_DISPLAY)
        return display?.rotation ?: android.view.Surface.ROTATION_0
    }

    LaunchedEffect(lifecycleOwner) {
        val cameraProvider = ProcessCameraProvider.getInstance(context).get()
        val initialRotation = getCurrentDisplayRotation()

        val preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .setTargetRotation(initialRotation)
            .build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
        previewUseCase = preview

        val recorderBuilder = Recorder.Builder().setQualitySelector(QualitySelector.from(Quality.HIGHEST))
        val videoCap = VideoCapture.Builder(recorderBuilder.build())
            .setTargetRotation(initialRotation)
            .build()
        videoCapture = videoCap

        val imgCap = ImageCapture.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .setTargetRotation(initialRotation)
            .build()
        imageCapture = imgCap

        val py = try { Python.getInstance() } catch (e: Exception) { null }
        val expModule = py?.getModule("exposure_analyzer")
        val compModule = py?.getModule("composition_analyzer")

        val exposureAnalyzer = expModule?.callAttr("ExposureAnalyzer")
        var compositionAnalyzer: PyObject? = null
        kotlinx.coroutines.withContext(Dispatchers.IO) {
            try {
                compositionAnalyzer = compModule?.callAttr("CompositionAnalyzer")
            } catch (e: Exception) {
                Log.e("Chaquopy", "Failed to init CompositionAnalyzer", e)
            }
        }

        val currentRotAfterLoad = getCurrentDisplayRotation()
        preview.targetRotation = currentRotAfterLoad
        videoCap.targetRotation = currentRotAfterLoad
        imgCap.targetRotation = currentRotAfterLoad

        val analyzerExecutor = Executors.newSingleThreadExecutor()
        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .setTargetRotation(currentRotAfterLoad)
            .build()
            .also { analysisUseCase ->
                imageAnalysisUseCase = analysisUseCase
                analysisUseCase.setAnalyzer(analyzerExecutor) { imageProxy ->
                    try {
                        val nv21Bytes = YuvConverter.imageProxyToNv21(imageProxy)
                        val rotation = imageProxy.imageInfo.rotationDegrees

                        if (exposureAnalyzer != null) {
                            val expResult: PyObject = exposureAnalyzer.callAttr("analyze_frame_nv21", nv21Bytes, imageProxy.width, imageProxy.height)
                            exposureFeedback = expResult.asMap().entries.associate { it.key.toString() to it.value.toString() }
                        }

                        if (compositionAnalyzer != null) {
                            val compResult: PyObject = compositionAnalyzer.callAttr("process_frame_nv21", nv21Bytes, imageProxy.width, imageProxy.height, rotation)
                            compositionFeedback = compResult.asMap().entries.associate { it.key.toString() to it.value.toString() }
                        }
                    } catch (e: Exception) {
                        Log.e("Chaquopy", "Computer Vision Error", e)
                    } finally {
                        imageProxy.close()
                    }
                }
            }

        try {
            cameraProvider.unbindAll()
            camera = cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imgCap, videoCap, imageAnalysis)
            awaitCancellation()
        } catch (exc: Exception) {
            Log.e("CameraX", "Binding failed", exc)
        } finally {
            cameraProvider.unbindAll()
            analyzerExecutor.shutdown()
        }
    }

    // Dynamically track all display rotation changes (including 90° and 180° landscape flips)
    DisposableEffect(previewUseCase, imageCapture, imageAnalysisUseCase, videoCapture) {
        val displayManager = context.getSystemService(android.content.Context.DISPLAY_SERVICE) as android.hardware.display.DisplayManager
        val displayListener = object : android.hardware.display.DisplayManager.DisplayListener {
            override fun onDisplayAdded(displayId: Int) {}
            override fun onDisplayRemoved(displayId: Int) {}
            override fun onDisplayChanged(displayId: Int) {
                if (displayId == android.view.Display.DEFAULT_DISPLAY) {
                    val rot = getCurrentDisplayRotation()
                    previewUseCase?.targetRotation = rot
                    imageCapture?.targetRotation = rot
                    imageAnalysisUseCase?.targetRotation = rot
                    videoCapture?.targetRotation = rot
                    previewView.post { previewView.requestLayout() }
                }
            }
        }
        displayManager.registerDisplayListener(displayListener, null)
        // Also sync immediately when use cases become non-null
        val rot = getCurrentDisplayRotation()
        previewUseCase?.targetRotation = rot
        imageCapture?.targetRotation = rot
        imageAnalysisUseCase?.targetRotation = rot
        videoCapture?.targetRotation = rot
        onDispose {
            displayManager.unregisterDisplayListener(displayListener)
        }
    }

    LaunchedEffect(isoValue, focusDistance, exposureTimeNs, wbValue, camera) {
        delay(150)
        camera?.let { cam ->
            val camera2Control = Camera2CameraControl.from(cam.cameraControl)

            val rGain = (1.0f + (wbValue - 2000) / 6000f * 2.0f).coerceIn(1.0f, 3.0f)
            val bGain = (1.0f + (8000 - wbValue) / 6000f * 2.0f).coerceIn(1.0f, 3.0f)
            val rggb = RggbChannelVector(rGain, 1.0f, 1.0f, bGain)

            val captureRequestOptions = CaptureRequestOptions.Builder()
                .setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                .setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, isoValue)
                .setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureTimeNs)
                .setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                .setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE, focusDistance)
                .setCaptureRequestOption(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_OFF)
                .setCaptureRequestOption(CaptureRequest.COLOR_CORRECTION_MODE, CaptureRequest.COLOR_CORRECTION_MODE_TRANSFORM_MATRIX)
                .setCaptureRequestOption(CaptureRequest.COLOR_CORRECTION_GAINS, rggb)
                .build()

            camera2Control.setCaptureRequestOptions(captureRequestOptions)
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

        Canvas(modifier = Modifier.fillMaxSize()) {
            val w = size.width
            val h = size.height
            val linePaint = Color.White.copy(alpha = 0.3f)

            // 1. Base Rule-of-Thirds Grid
            drawLine(linePaint, Offset(w / 3f, 0f), Offset(w / 3f, h), strokeWidth = 2f)
            drawLine(linePaint, Offset(w * 2f / 3f, 0f), Offset(w * 2f / 3f, h), strokeWidth = 2f)
            drawLine(linePaint, Offset(0f, h / 3f), Offset(w, h / 3f), strokeWidth = 2f)
            drawLine(linePaint, Offset(0f, h * 2f / 3f), Offset(w, h * 2f / 3f), strokeWidth = 2f)

            val isAligned = compositionFeedback["is_aligned"] == "True"
            val targetGreen = Color(0xFF00E676)
            val dashEffect = androidx.compose.ui.graphics.PathEffect.dashPathEffect(floatArrayOf(14f, 10f), 0f)

            // 2. Green Target Crosshairs (POINT) & Green Target Grid Lines (HLINE / VLINE)
            val targetsStr = compositionFeedback["targets_norm"] ?: ""
            if (targetsStr.isNotEmpty()) {
                targetsStr.split(";").forEach { item ->
                    val parts = item.split(",")
                    when (parts.firstOrNull()) {
                        "POINT" -> if (parts.size >= 5) {
                            val tx = (parts[1].toFloatOrNull() ?: 0.5f) * w
                            val ty = (parts[2].toFloatOrNull() ?: 0.5f) * h
                            val ax = (parts[3].toFloatOrNull() ?: 0.5f) * w
                            val ay = (parts[4].toFloatOrNull() ?: 0.5f) * h

                            // Dashed guiding vector from subject anchor to green target cross
                            if (!isAligned) {
                                drawLine(
                                    color = targetGreen.copy(alpha = 0.75f),
                                    start = Offset(ax, ay),
                                    end = Offset(tx, ty),
                                    strokeWidth = 3f,
                                    pathEffect = dashEffect
                                )
                            }
                            // Green Target Crosshair + Outer Ring
                            val crossRadius = 26f
                            drawLine(targetGreen, Offset(tx - crossRadius, ty), Offset(tx + crossRadius, ty), strokeWidth = 6f)
                            drawLine(targetGreen, Offset(tx, ty - crossRadius), Offset(tx, ty + crossRadius), strokeWidth = 6f)
                            drawCircle(
                                color = targetGreen,
                                radius = crossRadius * 0.75f,
                                center = Offset(tx, ty),
                                style = androidx.compose.ui.graphics.drawscope.Stroke(width = 3f)
                            )
                        }
                        "HLINE" -> if (parts.size >= 4) {
                            val ty = (parts[1].toFloatOrNull() ?: 0.3333f) * h
                            val ax = (parts[2].toFloatOrNull() ?: 0.5f) * w
                            val ay = (parts[3].toFloatOrNull() ?: 0.5f) * h

                            // Highlight entire horizontal grid line in bright Green
                            drawLine(targetGreen, Offset(0f, ty), Offset(w, ty), strokeWidth = 6f)
                            if (!isAligned) {
                                drawLine(
                                    color = targetGreen.copy(alpha = 0.75f),
                                    start = Offset(ax, ay),
                                    end = Offset(ax, ty),
                                    strokeWidth = 3f,
                                    pathEffect = dashEffect
                                )
                            }
                            // Small green cross at projection point on the line
                            drawLine(targetGreen, Offset(ax - 18f, ty), Offset(ax + 18f, ty), strokeWidth = 5f)
                            drawLine(targetGreen, Offset(ax, ty - 18f), Offset(ax, ty + 18f), strokeWidth = 5f)
                        }
                        "VLINE" -> if (parts.size >= 4) {
                            val tx = (parts[1].toFloatOrNull() ?: 0.3333f) * w
                            val ax = (parts[2].toFloatOrNull() ?: 0.5f) * w
                            val ay = (parts[3].toFloatOrNull() ?: 0.5f) * h

                            // Highlight entire vertical grid line in bright Green
                            drawLine(targetGreen, Offset(tx, 0f), Offset(tx, h), strokeWidth = 6f)
                            if (!isAligned) {
                                drawLine(
                                    color = targetGreen.copy(alpha = 0.75f),
                                    start = Offset(ax, ay),
                                    end = Offset(tx, ay),
                                    strokeWidth = 3f,
                                    pathEffect = dashEffect
                                )
                            }
                            // Small green cross at projection point on the line
                            drawLine(targetGreen, Offset(tx - 18f, ay), Offset(tx + 18f, ay), strokeWidth = 5f)
                            drawLine(targetGreen, Offset(tx, ay - 18f), Offset(tx, ay + 18f), strokeWidth = 5f)
                        }
                    }
                }
            }

            // 3. Salient Object Bounding Boxes & Center Anchor Dots
            val objectsStr = compositionFeedback["objects_norm"] ?: ""
            if (objectsStr.isNotEmpty()) {
                val objColor = if (isAligned) targetGreen else Color(0xFFFFBF00)
                objectsStr.split(";").forEach { item ->
                    val coords = item.split(",")
                    if (coords.size >= 6) {
                        val bx = (coords[0].toFloatOrNull() ?: 0f) * w
                        val by = (coords[1].toFloatOrNull() ?: 0f) * h
                        val bw = (coords[2].toFloatOrNull() ?: 0f) * w
                        val bh = (coords[3].toFloatOrNull() ?: 0f) * h
                        val ax = (coords[4].toFloatOrNull() ?: 0f) * w
                        val ay = (coords[5].toFloatOrNull() ?: 0f) * h

                        drawRect(
                            color = objColor,
                            topLeft = Offset(bx, by),
                            size = androidx.compose.ui.geometry.Size(bw, bh),
                            style = androidx.compose.ui.graphics.drawscope.Stroke(width = 4.5f)
                        )
                        drawCircle(color = objColor, radius = 8f, center = Offset(ax, ay))
                    }
                }
            }

            // 4. Face Bounding Boxes & Eye-Level Headroom Anchor Ticks
            val facesStr = compositionFeedback["faces_norm"] ?: ""
            if (facesStr.isNotEmpty()) {
                val faceColor = if (isAligned) targetGreen else Color.Cyan
                facesStr.split(";").forEach { item ->
                    val coords = item.split(",")
                    if (coords.size >= 6) {
                        val fx = (coords[0].toFloatOrNull() ?: 0f) * w
                        val fy = (coords[1].toFloatOrNull() ?: 0f) * h
                        val fw = (coords[2].toFloatOrNull() ?: 0f) * w
                        val fh = (coords[3].toFloatOrNull() ?: 0f) * h
                        val ax = (coords[4].toFloatOrNull() ?: 0f) * w
                        val ay = (coords[5].toFloatOrNull() ?: 0f) * h

                        drawRect(
                            color = faceColor,
                            topLeft = Offset(fx, fy),
                            size = androidx.compose.ui.geometry.Size(fw, fh),
                            style = androidx.compose.ui.graphics.drawscope.Stroke(width = 5f)
                        )
                        // Eye-level headroom anchor marker
                        drawLine(faceColor, Offset(ax - 20f, ay), Offset(ax + 20f, ay), strokeWidth = 4f)
                        drawCircle(color = faceColor, radius = 8f, center = Offset(ax, ay))
                    }
                }
            }

            // 5. Gravity-Driven Digital Horizon Level
            val display = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) context.display else (context.getSystemService(android.content.Context.WINDOW_SERVICE) as android.view.WindowManager).defaultDisplay
            val angleOffset = when(display?.rotation) { android.view.Surface.ROTATION_90 -> 90f; android.view.Surface.ROTATION_180 -> 180f; android.view.Surface.ROTATION_270 -> -90f; else -> 0f }
            val angle = rotationAngle.floatValue - angleOffset

            val normalized = (angle % 90 + 90) % 90
            val dist = minOf(normalized, 90 - normalized)

            rotate(-angle) {
                val levelColor = if (dist < 2f) Color.Green else Color.White.copy(alpha = 0.7f)
                drawLine(levelColor, Offset(w / 2 - 100f, h / 2), Offset(w / 2 + 100f, h / 2), strokeWidth = 4f)
                drawLine(levelColor, Offset(w / 2, h / 2 - 20f), Offset(w / 2, h / 2 + 20f), strokeWidth = 2f)
            }
        }

        Column(modifier = Modifier.align(Alignment.TopCenter).padding(top = 48.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (exposureFeedback.isNotEmpty()) {
                Column(modifier = Modifier.background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(8.dp)).padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(text = exposureFeedback["primary_action"] ?: "", color = when(exposureFeedback["primary_color"]) { "RED" -> Color.Red; "AMBER" -> Color(0xFFFFBF00); "GREEN" -> Color.Green; "CYAN" -> Color.Cyan; else -> Color.White }, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    val secondary = exposureFeedback["secondary_action"] ?: ""
                    if (secondary.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(text = secondary, color = when(exposureFeedback["secondary_color"]) { "CYAN" -> Color.Cyan; else -> Color.White }, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }

            if (compositionFeedback.isNotEmpty()) {
                val compText = compositionFeedback["feedback_text"] ?: ""
                if (compText.isNotEmpty()) {
                    val isAligned = compositionFeedback["is_aligned"] == "True"
                    Column(modifier = Modifier.background(Color(0xFF1E3A8A).copy(alpha = 0.7f), RoundedCornerShape(8.dp)).padding(12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(text = compText, color = if (isAligned) Color.Green else Color.White, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    }
                }
            }
        }

        Row(modifier = Modifier.fillMaxWidth().padding(top = 48.dp, start = 16.dp, end = 16.dp)) {
            IconButton(onClick = onNavigateBack, modifier = Modifier.background(Color.Black.copy(alpha = 0.4f), CircleShape)) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Color.White) }
        }

        val controlsModifier = if (isLandscape) Modifier.align(Alignment.CenterEnd).fillMaxHeight().background(Color.Black.copy(alpha = 0.5f)).padding(horizontal = 16.dp, vertical = 24.dp) else Modifier.align(Alignment.BottomCenter).fillMaxWidth().background(Color.Black.copy(alpha = 0.5f)).padding(bottom = 48.dp, top = 16.dp)

        Box(modifier = controlsModifier) {
            if (isLandscape) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.End, modifier = Modifier.fillMaxHeight()) {
                    AnimatedVisibility(visible = activeSetting != "NONE") {
                        val itemsList = when (activeSetting) { "WB" -> wbOptions; "FOCUS" -> focusOptions; "ISO" -> isoOptions; "SHUTTER" -> shutterOptions; else -> emptyList() }
                        val activeState = listStates[activeSetting]
                        if (activeState != null && itemsList.isNotEmpty()) {
                            key(activeSetting) { ValueWheel(items = itemsList, selectedValue = when (activeSetting) { "WB" -> wbValue; "FOCUS" -> focusDistance; "ISO" -> isoValue; "SHUTTER" -> exposureTimeNs; else -> null }, onValueChange = { when (activeSetting) { "WB" -> wbValue = it as Int; "FOCUS" -> focusDistance = it as Float; "ISO" -> isoValue = it as Int; "SHUTTER" -> exposureTimeNs = it as Long } }, isLandscape = true, listState = activeState) }
                        }
                    }
                    Spacer(modifier = Modifier.width(24.dp))
                    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.SpaceEvenly, modifier = Modifier.fillMaxHeight()) {
                        SettingToggle("WB", "${wbValue}K", activeSetting == "WB") { activeSetting = if (activeSetting == "WB") "NONE" else "WB" }
                        SettingToggle("Focus", if (focusDistance == 0f) "Inf" else if (focusDistance == 10f) "Macro" else "MF", activeSetting == "FOCUS") { activeSetting = if (activeSetting == "FOCUS") "NONE" else "FOCUS" }
                        SettingToggle("ISO", "$isoValue", activeSetting == "ISO") { activeSetting = if (activeSetting == "ISO") "NONE" else "ISO" }
                        SettingToggle("S", shutterOptions.find { it.first == exposureTimeNs }?.second ?: "", activeSetting == "SHUTTER") { activeSetting = if (activeSetting == "SHUTTER") "NONE" else "SHUTTER" }
                    }
                    Spacer(modifier = Modifier.width(32.dp))
                    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.SpaceEvenly, modifier = Modifier.fillMaxHeight()) { CaptureButtons(context, imageCapture, videoCapture, recording, isVideoPaused, { recording = it }, { isVideoPaused = it }) }
                }
            } else {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    AnimatedVisibility(visible = activeSetting != "NONE") {
                        val itemsList = when (activeSetting) { "WB" -> wbOptions; "FOCUS" -> focusOptions; "ISO" -> isoOptions; "SHUTTER" -> shutterOptions; else -> emptyList() }
                        val activeState = listStates[activeSetting]
                        if (activeState != null && itemsList.isNotEmpty()) {
                            key(activeSetting) { ValueWheel(items = itemsList, selectedValue = when (activeSetting) { "WB" -> wbValue; "FOCUS" -> focusDistance; "ISO" -> isoValue; "SHUTTER" -> exposureTimeNs; else -> null }, onValueChange = { when (activeSetting) { "WB" -> wbValue = it as Int; "FOCUS" -> focusDistance = it as Float; "ISO" -> isoValue = it as Int; "SHUTTER" -> exposureTimeNs = it as Long } }, isLandscape = false, listState = activeState) }
                        }
                    }
                    Spacer(modifier = Modifier.height(16.dp))
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly, verticalAlignment = Alignment.CenterVertically) {
                        SettingToggle("WB", "${wbValue}K", activeSetting == "WB") { activeSetting = if (activeSetting == "WB") "NONE" else "WB" }
                        SettingToggle("Focus", if (focusDistance == 0f) "Inf" else if (focusDistance == 10f) "Macro" else "MF", activeSetting == "FOCUS") { activeSetting = if (activeSetting == "FOCUS") "NONE" else "FOCUS" }
                        SettingToggle("ISO", "$isoValue", activeSetting == "ISO") { activeSetting = if (activeSetting == "ISO") "NONE" else "ISO" }
                        SettingToggle("S", shutterOptions.find { it.first == exposureTimeNs }?.second ?: "", activeSetting == "SHUTTER") { activeSetting = if (activeSetting == "SHUTTER") "NONE" else "SHUTTER" }
                    }
                    Spacer(modifier = Modifier.height(16.dp))
                    Row(modifier = Modifier.fillMaxWidth().padding(horizontal = 32.dp), horizontalArrangement = Arrangement.SpaceEvenly, verticalAlignment = Alignment.CenterVertically) { CaptureButtons(context, imageCapture, videoCapture, recording, isVideoPaused, { recording = it }, { isVideoPaused = it }) }
                }
            }
        }
    }
}

@Composable
fun SettingToggle(label: String, currentValueStr: String, isSelected: Boolean, onClick: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.clickable(onClick = onClick).padding(8.dp)) {
        Text(label, color = if (isSelected) Color.Yellow else Color.White, style = MaterialTheme.typography.labelSmall)
        Spacer(modifier = Modifier.height(2.dp))
        Text(currentValueStr, color = if (isSelected) Color.Yellow else Color.White, style = MaterialTheme.typography.titleSmall, fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal)
    }
}

@OptIn(androidx.compose.foundation.ExperimentalFoundationApi::class)
@Composable
fun <T> ValueWheel(items: List<Pair<T, String>>, selectedValue: T?, onValueChange: (T) -> Unit, isLandscape: Boolean, listState: LazyListState) {
    val snapBehavior = rememberSnapFlingBehavior(lazyListState = listState)
    val hapticFeedback = LocalHapticFeedback.current
    val itemSizeDp = 48.dp

    val centerItemIndex by remember(listState, isLandscape) {
        derivedStateOf {
            val layoutInfo = listState.layoutInfo
            val visibleItemsInfo = layoutInfo.visibleItemsInfo
            if (visibleItemsInfo.isEmpty()) return@derivedStateOf -1
            val viewportCenter = (layoutInfo.viewportStartOffset + layoutInfo.viewportEndOffset) / 2
            visibleItemsInfo.minByOrNull { Math.abs(it.offset + (it.size / 2) - viewportCenter) }?.index ?: -1
        }
    }

    LaunchedEffect(centerItemIndex) {
        if (listState.isScrollInProgress && centerItemIndex in items.indices) {
            val newValue = items[centerItemIndex].first
            if (newValue != selectedValue) {
                hapticFeedback.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                onValueChange(newValue)
            }
        }
    }

    BoxWithConstraints(modifier = Modifier.run { if (isLandscape) width(80.dp).fillMaxHeight() else fillMaxWidth().height(80.dp) }) {
        val halfSize = if (isLandscape) maxHeight / 2 else maxWidth / 2
        val paddingDp = halfSize - (itemSizeDp / 2)
        val isFocusWheel = items.any { it.second == "Macro" }

        if (isLandscape) {
            LazyColumn(state = listState, flingBehavior = snapBehavior, modifier = Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally, contentPadding = PaddingValues(vertical = paddingDp)) {
                itemsIndexed(items) { index, item ->
                    val isCenter = index == centerItemIndex
                    val isTick = item.second == "|"
                    val color by animateColorAsState(if (isCenter) Color.Yellow else if (isTick) Color.White.copy(alpha = 0.5f) else Color.White, label = "color")
                    val scale by animateFloatAsState(if (isCenter) 1.2f else 1.0f, label = "scale")
                    Box(modifier = Modifier.height(itemSizeDp).fillMaxWidth(), contentAlignment = Alignment.Center) { Text(item.second, color = color, fontWeight = if (isCenter && !isTick) FontWeight.Bold else FontWeight.Normal, style = MaterialTheme.typography.labelMedium, modifier = Modifier.scale(scale)) }
                }
            }
            if (isFocusWheel) {
                Text("Inf", color = Color.White.copy(alpha = 0.3f), style = MaterialTheme.typography.labelSmall, modifier = Modifier.align(Alignment.TopCenter).padding(top = 16.dp))
                Text("Macro", color = Color.White.copy(alpha = 0.3f), style = MaterialTheme.typography.labelSmall, modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 16.dp))
            }
            Canvas(modifier = Modifier.matchParentSize()) { drawLine(Color.Yellow, Offset(size.width - 15f, size.height / 2), Offset(size.width, size.height / 2), strokeWidth = 8f) }
        } else {
            LazyRow(state = listState, flingBehavior = snapBehavior, modifier = Modifier.fillMaxSize(), verticalAlignment = Alignment.CenterVertically, contentPadding = PaddingValues(horizontal = paddingDp)) {
                itemsIndexed(items) { index, item ->
                    val isCenter = index == centerItemIndex
                    val isTick = item.second == "|"
                    val color by animateColorAsState(if (isCenter) Color.Yellow else if (isTick) Color.White.copy(alpha = 0.5f) else Color.White, label = "color")
                    val scale by animateFloatAsState(if (isCenter) 1.2f else 1.0f, label = "scale")
                    Box(modifier = Modifier.width(itemSizeDp).fillMaxHeight(), contentAlignment = Alignment.Center) { Text(item.second, color = color, fontWeight = if (isCenter && !isTick) FontWeight.Bold else FontWeight.Normal, style = MaterialTheme.typography.labelMedium, modifier = Modifier.scale(scale)) }
                }
            }
            if (isFocusWheel) {
                Text("Inf", color = Color.White.copy(alpha = 0.3f), style = MaterialTheme.typography.labelSmall, modifier = Modifier.align(Alignment.CenterStart).padding(start = 16.dp))
                Text("Macro", color = Color.White.copy(alpha = 0.3f), style = MaterialTheme.typography.labelSmall, modifier = Modifier.align(Alignment.CenterEnd).padding(end = 16.dp))
            }
            Canvas(modifier = Modifier.matchParentSize()) { drawLine(Color.Yellow, Offset(size.width / 2, size.height - 15f), Offset(size.width / 2, size.height), strokeWidth = 8f) }
        }
    }
}

@SuppressLint("MissingPermission")
@Composable
fun CaptureButtons(context: android.content.Context, imageCapture: ImageCapture?, videoCapture: VideoCapture<Recorder>?, recording: Recording?, isVideoPaused: Boolean, onRecordingChange: (Recording?) -> Unit, onPauseChange: (Boolean) -> Unit) {
    Box(
        modifier = Modifier.size(50.dp).clip(CircleShape).background(if (recording != null) Color.Red else Color.White.copy(alpha = 0.2f)).clickable {
            if (recording != null) { recording.stop(); onRecordingChange(null) } else {
                val outputFile = File(context.getExternalFilesDir(null), "VIDEO_${System.currentTimeMillis()}.mp4")
                val options = FileOutputOptions.Builder(outputFile).build()
                val newRecording = videoCapture?.output?.prepareRecording(context, options)?.withAudioEnabled()?.start(ContextCompat.getMainExecutor(context)) { event ->
                    if (event is VideoRecordEvent.Finalize) Toast.makeText(context, "Video Saved", Toast.LENGTH_SHORT).show()
                }
                onRecordingChange(newRecording); onPauseChange(false)
            }
        }, contentAlignment = Alignment.Center
    ) {
        if (recording != null) { Box(modifier = Modifier.size(20.dp).clip(RoundedCornerShape(4.dp)).background(Color.White)) } else { Icon(Icons.Default.Videocam, contentDescription = "Video", tint = Color.White) }
    }

    IconButton(
        onClick = {
            val outputFile = File(context.getExternalFilesDir(null), "PHOTO_${System.currentTimeMillis()}.jpg")
            val outputOptions = ImageCapture.OutputFileOptions.Builder(outputFile).build()
            imageCapture?.takePicture(outputOptions, ContextCompat.getMainExecutor(context), object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) { Toast.makeText(context, "Photo Saved", Toast.LENGTH_SHORT).show() }
                override fun onError(exc: ImageCaptureException) { Log.e("CameraX", "Photo capture failed", exc) }
            })
        },
        modifier = Modifier.size(72.dp).background(Color.White.copy(alpha = 0.5f), CircleShape)
    ) { Box(modifier = Modifier.size(56.dp).clip(CircleShape).background(Color.White)) }

    IconButton(
        onClick = { if (recording != null) { if (isVideoPaused) { recording.resume(); onPauseChange(false) } else { recording.pause(); onPauseChange(true) } } },
        enabled = recording != null,
        modifier = Modifier.size(50.dp).background(if (recording != null) Color.White.copy(alpha = 0.2f) else Color.Transparent, CircleShape)
    ) { if (recording != null) Icon(if (isVideoPaused) Icons.Default.PlayArrow else Icons.Default.Pause, contentDescription = "Pause", tint = Color.White) }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AudioModeScreen(onNavigateBack: () -> Unit) {
    val context = LocalContext.current
    var hasMicPermission by remember { mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { hasMicPermission = it }

    LaunchedEffect(Unit) { if (!hasMicPermission) permissionLauncher.launch(Manifest.permission.RECORD_AUDIO) }
    if (!hasMicPermission) { PermissionDeniedScreen("Microphone", onNavigateBack) { permissionLauncher.launch(Manifest.permission.RECORD_AUDIO) }; return }

    Scaffold(topBar = { TopAppBar(title = { Text("Acoustic Module") }, navigationIcon = { IconButton(onClick = onNavigateBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") } }) }) { paddingValues ->
        Column(modifier = Modifier.padding(paddingValues).fillMaxSize()) { RecordTab(context) }
    }
}

@SuppressLint("MissingPermission")
@Composable
fun RecordTab(context: android.content.Context) {
    var isRecording by remember { mutableStateOf(false) }
    var recordTime by remember { mutableIntStateOf(0) }
    var recordingJob by remember { mutableStateOf<Job?>(null) }
    val scope = rememberCoroutineScope()

    // Real-time acoustic diagnostic states
    var primaryAudioAction by remember { mutableStateOf("Ready to monitor acoustic environment") }
    var primaryAudioColor by remember { mutableStateOf("WHITE") }
    var secondaryAudioAction by remember { mutableStateOf("") }
    var liveRmsDb by remember { mutableFloatStateOf(-60f) }
    var livePeakDb by remember { mutableFloatStateOf(-60f) }
    var liveSnrDb by remember { mutableFloatStateOf(0f) }

    LaunchedEffect(isRecording) { while (isRecording) { delay(1000); recordTime++ } }
    DisposableEffect(Unit) { onDispose { recordingJob?.cancel() } }

    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val scale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = if (isRecording) 1.2f else 1f,
        animationSpec = infiniteRepeatable(animation = tween(800, easing = LinearEasing), repeatMode = RepeatMode.Reverse),
        label = "pulse_anim"
    )

    Column(
        modifier = Modifier.fillMaxSize().padding(horizontal = 20.dp),
        verticalArrangement = Arrangement.SpaceBetween,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Top Real-Time Acoustic Diagnostic Banner (matching VideoModeScreen style)
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 16.dp)
                .background(Color.Black.copy(alpha = 0.75f), RoundedCornerShape(12.dp))
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = primaryAudioAction,
                color = when (primaryAudioColor) {
                    "RED" -> Color.Red
                    "AMBER" -> Color(0xFFFFBF00)
                    "GREEN" -> Color.Green
                    else -> Color.White
                },
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )
            if (secondaryAudioAction.isNotEmpty()) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = secondaryAudioAction,
                    color = Color.Cyan,
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center
                )
            }
            if (isRecording) {
                Spacer(modifier = Modifier.height(10.dp))
                Text(
                    text = String.format(Locale.US, "[Peak: %.1f dBFS | RMS: %.1f dBFS | Est. SNR: %.1f dB]", livePeakDb, liveRmsDb, liveSnrDb),
                    color = Color.White.copy(alpha = 0.8f),
                    style = MaterialTheme.typography.labelSmall
                )
                Spacer(modifier = Modifier.height(6.dp))
                // Live dBFS Meter Bar (-60 dBFS to 0 dBFS)
                val meterFraction = ((liveRmsDb + 60f) / 60f).coerceIn(0f, 1f)
                LinearProgressIndicator(
                    progress = { meterFraction },
                    modifier = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)),
                    color = when {
                        livePeakDb > -1.0f -> Color.Red
                        liveRmsDb < -35.0f -> Color(0xFFFFBF00)
                        else -> Color.Green
                    },
                    trackColor = Color.DarkGray
                )
            }
        }

        // Center Timer & Record Button
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = String.format(Locale.US, "%02d:%02d", recordTime / 60, recordTime % 60),
                style = MaterialTheme.typography.displayLarge,
                fontWeight = FontWeight.Light,
                color = if (isRecording) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface
            )
            Spacer(modifier = Modifier.height(36.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(24.dp), verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(100.dp)
                        .scale(scale)
                        .clip(CircleShape)
                        .background(if (isRecording) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary)
                        .clickable {
                            if (!isRecording) {
                                isRecording = true
                                val outputFile = File(context.getExternalFilesDir(null), "AUDIO_${System.currentTimeMillis()}.wav")
                                recordingJob = scope.launch(Dispatchers.IO) {
                                    val sampleRate = 44100
                                    val channelConfig = AudioFormat.CHANNEL_IN_MONO
                                    val audioFormat = AudioFormat.ENCODING_PCM_16BIT
                                    val minBuf = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
                                    // Use ~50ms chunks (4410 bytes = 2205 16-bit samples) for responsive real-time metering
                                    val bufferSize = maxOf(minBuf, 4410)
                                    val audioRecord = AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate, channelConfig, audioFormat, bufferSize)

                                    if (audioRecord.state != AudioRecord.STATE_INITIALIZED) return@launch
                                    val out = FileOutputStream(outputFile)
                                    out.write(ByteArray(44)) // Placeholder for WAV header
                                    val buffer = ByteArray(bufferSize)
                                    audioRecord.startRecording()
                                    var totalAudioLen = 0L

                                    // Rolling buffers for real-time SNR & Reverberation tail diagnostics
                                    val rmsHistory = ArrayDeque<Float>() // Last 50 frames (~2.5 seconds)
                                    val decaySlopes = ArrayDeque<Float>() // Post-peak 150ms decay drops
                                    var holdClipFrames = 0

                                    try {
                                        while (isActive) {
                                            val read = audioRecord.read(buffer, 0, bufferSize)
                                            if (read > 0) {
                                                out.write(buffer, 0, read)
                                                totalAudioLen += read

                                                // --- REAL-TIME ACOUSTIC DIAGNOSTICS ---
                                                val nSamples = read / 2
                                                if (nSamples > 0) {
                                                    var sumSq = 0.0
                                                    var maxAbs = 0
                                                    var i = 0
                                                    while (i < read - 1) {
                                                        val s = ((buffer[i + 1].toInt() shl 8) or (buffer[i].toInt() and 0xFF)).toShort().toInt()
                                                        val absS = if (s < 0) -s else s
                                                        if (absS > maxAbs) maxAbs = absS
                                                        val norm = s / 32768.0
                                                        sumSq += norm * norm
                                                        i += 2
                                                    }
                                                    val rmsLin = kotlin.math.sqrt(sumSq / nSamples).coerceAtLeast(1e-6)
                                                    val peakLin = (maxAbs / 32768.0).coerceAtLeast(1e-6)
                                                    val frameRmsDb = (20.0 * kotlin.math.log10(rmsLin)).toFloat()
                                                    val framePeakDb = (20.0 * kotlin.math.log10(peakLin)).toFloat()

                                                    rmsHistory.addLast(frameRmsDb)
                                                    if (rmsHistory.size > 50) rmsHistory.removeFirst()

                                                    val sortedRms = rmsHistory.sorted()
                                                    val p10Noise = sortedRms[(sortedRms.size * 0.10).toInt().coerceIn(0, sortedRms.size - 1)]
                                                    val p90Speech = sortedRms[(sortedRms.size * 0.90).toInt().coerceIn(0, sortedRms.size - 1)]
                                                    val estSnr = (p90Speech - p10Noise).coerceAtLeast(0f)

                                                    // Check 3-frame (150ms) post-syllable decay rate for room reverberation
                                                    if (rmsHistory.size >= 6) {
                                                        val histList = rmsHistory.toList()
                                                        val nH = histList.size
                                                        val prevPeak = histList[nH - 4]
                                                        val currTail = histList[nH - 1]
                                                        // If 3 frames ago was an active syllable peak and speaker stopped
                                                        if (prevPeak > p90Speech - 3.0f && currTail < prevPeak - 2.0f && estSnr >= 12.0f) {
                                                            val dropDb = prevPeak - currTail
                                                            decaySlopes.addLast(dropDb)
                                                            if (decaySlopes.size > 8) decaySlopes.removeFirst()
                                                        }
                                                    }
                                                    val avgDecayDrop = if (decaySlopes.size >= 3) decaySlopes.average().toFloat() else 15.0f

                                                    if (framePeakDb > -0.6f) holdClipFrames = 12
                                                    else if (holdClipFrames > 0) holdClipFrames--

                                                    val (pAction, pColor) = when {
                                                        holdClipFrames > 0 ->
                                                            "SEVERE CLIPPING. Move microphone further away." to "RED"
                                                        rmsHistory.size >= 15 && p10Noise > -36.0f ->
                                                            "HIGH BACKGROUND NOISE. Move to a quieter location." to "AMBER"
                                                        rmsHistory.size >= 20 && estSnr < 11.0f && p90Speech > -42.0f ->
                                                            "LOW SNR (< 11 dB). Too much ambient noise." to "AMBER"
                                                        rmsHistory.size >= 20 && p90Speech < -34.0f ->
                                                            "WEAK VOICE SIGNAL. Move microphone closer to speaker." to "AMBER"
                                                        else ->
                                                            "Acoustic Level Locked." to "GREEN"
                                                    }

                                                    val sAction = if (decaySlopes.size >= 3 && avgDecayDrop < 8.5f && estSnr >= 12.0f) {
                                                        "HIGH ROOM REVERBERATION. Acoustic tail detected; move closer."
                                                    } else {
                                                        ""
                                                    }

                                                    liveRmsDb = frameRmsDb
                                                    livePeakDb = framePeakDb
                                                    liveSnrDb = estSnr
                                                    primaryAudioAction = pAction
                                                    primaryAudioColor = pColor
                                                    secondaryAudioAction = sAction
                                                }
                                            }
                                        }
                                    } finally {
                                        audioRecord.stop()
                                        audioRecord.release()

                                        val totalDataLen = totalAudioLen + 36
                                        val byteRate = sampleRate * 2

                                        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
                                            put("RIFF".toByteArray(Charsets.US_ASCII))
                                            putInt(totalDataLen.toInt())
                                            put("WAVE".toByteArray(Charsets.US_ASCII))
                                            put("fmt ".toByteArray(Charsets.US_ASCII))
                                            putInt(16)
                                            putShort(1.toShort())
                                            putShort(1.toShort())
                                            putInt(sampleRate)
                                            putInt(byteRate)
                                            putShort(2.toShort())
                                            putShort(16.toShort())
                                            put("data".toByteArray(Charsets.US_ASCII))
                                            putInt(totalAudioLen.toInt())
                                        }.array()

                                        val randomAccessFile = java.io.RandomAccessFile(outputFile, "rw")
                                        randomAccessFile.seek(0)
                                        randomAccessFile.write(header)
                                        randomAccessFile.close()
                                        out.close()

                                        kotlinx.coroutines.withContext(Dispatchers.Main) {
                                            primaryAudioAction = "Recording saved. Ready for on-device AI analysis."
                                            primaryAudioColor = "WHITE"
                                            secondaryAudioAction = ""
                                            Toast.makeText(context, "Audio Saved (.wav)", Toast.LENGTH_SHORT).show()
                                        }
                                    }
                                }
                            } else {
                                isRecording = false
                                recordTime = 0
                                recordingJob?.cancel()
                                recordingJob = null
                            }
                        },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = if (!isRecording) Icons.Default.Mic else Icons.Default.Stop,
                        contentDescription = "Record",
                        tint = Color.White,
                        modifier = Modifier.size(40.dp)
                    )
                }
            }
            Spacer(modifier = Modifier.height(24.dp))
            Text(
                text = if (isRecording) "Recording & monitoring acoustics..." else "Tap microphone to start recording",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        Spacer(modifier = Modifier.height(24.dp))
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GalleryScreen(onNavigateBack: () -> Unit, onNavigateToTranscription: (String) -> Unit, onNavigateToImageViewer: (String) -> Unit) {
    val context = LocalContext.current
    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf("Images & Videos", "Audio")

    Scaffold(topBar = { TopAppBar(title = { Text("Gallery") }, navigationIcon = { IconButton(onClick = onNavigateBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") } }) }) { paddingValues ->
        Column(modifier = Modifier.padding(paddingValues).fillMaxSize()) {
            PrimaryTabRow(selectedTabIndex = selectedTab) { tabs.forEachIndexed { index, title -> Tab(selected = selectedTab == index, onClick = { selectedTab = index }, text = { Text(title, fontWeight = FontWeight.SemiBold) }) } }
            AnimatedContent(targetState = selectedTab, label = "GalleryTabs") { tab ->
                when (tab) {
                    0 -> VisualLibraryTab(context, onNavigateToTranscription, onNavigateToImageViewer)
                    1 -> AudioLibraryTab(context, onNavigateToTranscription)
                }
            }
        }
    }
}

@Composable
fun VisualLibraryTab(context: android.content.Context, onNavigateToTranscription: (String) -> Unit, onNavigateToImageViewer: (String) -> Unit) {
    val mediaDir = context.getExternalFilesDir(null)
    val visuals = mediaDir?.listFiles { file -> file.extension == "jpg" || file.extension == "mp4" }?.sortedByDescending { it.lastModified() }?.map { it.name } ?: emptyList()

    if (visuals.isEmpty()) { Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Text("No photos or videos found.") } } else {
        LazyColumn(modifier = Modifier.fillMaxSize().padding(16.dp)) {
            items(visuals) { fileName ->
                val isVideo = fileName.endsWith(".mp4")
                ElevatedCard(
                    modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp),
                    onClick = {
                        val path = File(mediaDir, fileName).absolutePath
                        if (isVideo) onNavigateToTranscription(path) else onNavigateToImageViewer(path)
                    },
                    colors = CardDefaults.elevatedCardColors(containerColor = MaterialTheme.colorScheme.surface)
                ) {
                    Row(modifier = Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(if (isVideo) Icons.Default.Videocam else Icons.Default.Image, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(modifier = Modifier.width(16.dp))
                        Column { Text(fileName, style = MaterialTheme.typography.titleMedium) }
                    }
                }
            }
        }
    }
}

@Composable
fun AudioLibraryTab(context: android.content.Context, onNavigateToTranscription: (String) -> Unit) {
    val audioDir = context.getExternalFilesDir(null)
    val recordings = audioDir?.listFiles { file -> file.extension == "wav" }?.sortedByDescending { it.lastModified() }?.map { it.name } ?: emptyList()

    if (recordings.isEmpty()) { Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Text("No audio recordings found.") } } else {
        LazyColumn(modifier = Modifier.fillMaxSize().padding(16.dp)) {
            items(recordings) { fileName ->
                ElevatedCard(modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp), onClick = { onNavigateToTranscription(File(audioDir, fileName).absolutePath) }, colors = CardDefaults.elevatedCardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                    Row(modifier = Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.AudioFile, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(modifier = Modifier.width(16.dp))
                        Column { Text(fileName, style = MaterialTheme.typography.titleMedium) }
                    }
                }
            }
        }
    }
}

// =========================================================================
// ML TRANSCRIPTION & PRESETS SCREEN
// =========================================================================

@androidx.annotation.OptIn(androidx.media3.common.util.UnstableApi::class)
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TranscriptionScreen(filePath: String, onNavigateBack: () -> Unit) {
    val context = LocalContext.current
    val isAudio = filePath.endsWith(".wav")
    val exoPlayer = remember {
        ExoPlayer.Builder(context).build().apply {
            setMediaItem(MediaItem.fromUri(filePath))
            prepare()
        }
    }

    DisposableEffect(Unit) { onDispose { exoPlayer.release() } }

    var showTranscription by remember { mutableStateOf(true) }
    var showSettings by remember { mutableStateOf(true) }

    // PyTorch & Limiter States
    var effectPresets by remember { mutableStateOf<Map<String, String>?>(null) }
    var predictedIndices by remember { mutableStateOf<IntArray?>(null) }
    var isAnalyzing by remember { mutableStateOf(false) }
    var mlErrorMessage by remember { mutableStateOf<String?>(null) }

    // On-Device DSP Rendering & A/B Playback States
    var isRenderingAudio by remember { mutableStateOf(false) }
    var enhancedFilePath by remember { mutableStateOf<String?>(null) }
    var isPlayingEnhanced by remember { mutableStateOf(false) }

    // Vosk States
    var transcriptionText by remember { mutableStateOf("") }
    var isTranscribing by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(if (isAudio) "Audio Analytics" else "Video Analytics") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(modifier = Modifier.fillMaxSize().padding(paddingValues).padding(16.dp)) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(if (isAudio) 80.dp else 200.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(Color.Black)
            ) {
                AndroidView(
                    factory = { ctx ->
                        PlayerView(ctx).apply {
                            player = exoPlayer
                            if (isAudio) {
                                setShowBuffering(PlayerView.SHOW_BUFFERING_NEVER)
                                controllerAutoShow = true
                                controllerShowTimeoutMs = -1
                                useArtwork = false
                            }
                        }
                    },
                    modifier = Modifier.fillMaxSize()
                )
            }

            // A/B Playback Toggle when Enhanced Audio has been rendered
            if (enhancedFilePath != null) {
                Spacer(modifier = Modifier.height(10.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    FilledTonalButton(
                        onClick = {
                            if (isPlayingEnhanced) {
                                isPlayingEnhanced = false
                                exoPlayer.setMediaItem(MediaItem.fromUri(filePath))
                                exoPlayer.prepare()
                                exoPlayer.play()
                            }
                        },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.filledTonalButtonColors(
                            containerColor = if (!isPlayingEnhanced) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
                        )
                    ) {
                        Text("Original (Raw)", fontWeight = if (!isPlayingEnhanced) FontWeight.Bold else FontWeight.Normal)
                    }
                    FilledTonalButton(
                        onClick = {
                            if (!isPlayingEnhanced) {
                                isPlayingEnhanced = true
                                exoPlayer.setMediaItem(MediaItem.fromUri(enhancedFilePath!!))
                                exoPlayer.prepare()
                                exoPlayer.play()
                            }
                        },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.filledTonalButtonColors(
                            containerColor = if (isPlayingEnhanced) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
                        )
                    ) {
                        Text("Enhanced (AI DSP)", fontWeight = if (isPlayingEnhanced) FontWeight.Bold else FontWeight.Normal)
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            LazyColumn(modifier = Modifier.fillMaxSize()) {
                item {
                    val activeFileName = if (isPlayingEnhanced && enhancedFilePath != null) {
                        File(enhancedFilePath!!).name
                    } else {
                        File(filePath).name
                    }
                    Text("Playing: $activeFileName", style = MaterialTheme.typography.bodyMedium)
                    Spacer(modifier = Modifier.height(16.dp))
                }

                if (!isAudio) {
                    item {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(8.dp))
                                .padding(16.dp)
                        ) {
                            Text(
                                "Transcription and Acoustic Preset Recommendations are only available for .wav audio recordings. Raw audio files are never altered.",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                } else {
                    item {
                        ElevatedCard(
                            onClick = { showTranscription = !showTranscription },
                            modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp)
                        ) {
                            Row(
                                modifier = Modifier.padding(16.dp).fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text("Transcription", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.primary)
                                Icon(if (showTranscription) Icons.Default.ExpandLess else Icons.Default.ExpandMore, contentDescription = null)
                            }
                        }
                    }
                    item {
                        AnimatedVisibility(visible = showTranscription) {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(8.dp))
                                    .padding(16.dp)
                            ) {
                                if (transcriptionText.isNotEmpty()) {
                                    Text(transcriptionText, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                } else {
                                    Button(
                                        onClick = {
                                            isTranscribing = true
                                            loadVoskModel(
                                                context,
                                                { model ->
                                                    kotlinx.coroutines.CoroutineScope(Dispatchers.IO).launch {
                                                        try {
                                                            val wavFile = File(filePath)
                                                            val audioDurationSeconds = (wavFile.length() - 44) / 88200.0
                                                            val voskStartTime = System.currentTimeMillis()

                                                            val recognizer = Recognizer(model, 44100f)
                                                            val fis = FileInputStream(wavFile)
                                                            fis.skip(44)

                                                            val buffer = ByteArray(4096)
                                                            var bytesRead: Int
                                                            val fullText = java.lang.StringBuilder()

                                                            while (fis.read(buffer).also { bytesRead = it } != -1) {
                                                                if (recognizer.acceptWaveForm(buffer, bytesRead)) {
                                                                    val result = JSONObject(recognizer.result)
                                                                    val text = result.optString("text", "")
                                                                    if (text.isNotEmpty()) fullText.append(text).append(" ")
                                                                }
                                                            }

                                                            val finalResult = JSONObject(recognizer.finalResult)
                                                            val finalText = finalResult.optString("text", "")
                                                            if (finalText.isNotEmpty()) fullText.append(finalText)

                                                            fis.close()

                                                            val voskProcessingTime = (System.currentTimeMillis() - voskStartTime) / 1000.0
                                                            val voskRtf = voskProcessingTime / audioDurationSeconds
                                                            Log.d("ThesisBenchmark", "VOSK RTF: $voskRtf (Audio: ${audioDurationSeconds}s, Time: ${voskProcessingTime}s)")

                                                            kotlinx.coroutines.withContext(Dispatchers.Main) {
                                                                transcriptionText = fullText.toString().ifEmpty { "No recognizable speech detected." }
                                                                isTranscribing = false
                                                            }
                                                        } catch (e: Exception) {
                                                            Log.e("VOSK", "Error during transcription", e)
                                                            kotlinx.coroutines.withContext(Dispatchers.Main) {
                                                                transcriptionText = "Error occurred during transcription."
                                                                isTranscribing = false
                                                            }
                                                        }
                                                    }
                                                },
                                                { exception ->
                                                    Log.e("VOSK", "Failed to load model", exception)
                                                    isTranscribing = false
                                                    transcriptionText = "Failed to load Vosk model from assets."
                                                }
                                            )
                                        },
                                        modifier = Modifier.fillMaxWidth(),
                                        enabled = !isTranscribing
                                    ) {
                                        Text(if (isTranscribing) "Transcribing Audio..." else "Generate Transcription")
                                    }
                                }
                            }
                        }
                        Spacer(modifier = Modifier.height(16.dp))
                    }
                    item {
                        ElevatedCard(
                            onClick = { showSettings = !showSettings },
                            modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp)
                        ) {
                            Row(
                                modifier = Modifier.padding(16.dp).fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text("Acoustic Preset Recommendations", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.primary)
                                Icon(if (showSettings) Icons.Default.ExpandLess else Icons.Default.ExpandMore, contentDescription = null)
                            }
                        }
                    }
                    item {
                        AnimatedVisibility(visible = showSettings) {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(8.dp))
                                    .padding(16.dp)
                            ) {
                                if (mlErrorMessage != null) {
                                    Column {
                                        Text("Analysis Error:", color = MaterialTheme.colorScheme.error, fontWeight = FontWeight.Bold)
                                        Text(mlErrorMessage!!, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                                        Spacer(modifier = Modifier.height(8.dp))
                                        Button(onClick = { mlErrorMessage = null }) { Text("Dismiss & Retry") }
                                    }
                                } else if (effectPresets != null) {
                                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                        Text(
                                            "White-Box DAW Parameters (or render directly on device):",
                                            style = MaterialTheme.typography.labelMedium,
                                            color = MaterialTheme.colorScheme.primary
                                        )
                                        Text("Denoise: ${effectPresets?.get("denoise") ?: ""}")
                                        Text("Dynamic: ${effectPresets?.get("compressor") ?: ""}")
                                        Text("EQ: ${effectPresets?.get("eq") ?: ""}")
                                        Text("De-Esser: ${effectPresets?.get("deesser") ?: ""}")
                                        Text("Space: ${effectPresets?.get("reverb") ?: ""}")
                                        Text("Limiter: ${effectPresets?.get("limiter") ?: ""}")

                                        Spacer(modifier = Modifier.height(8.dp))

                                        Button(
                                            onClick = {
                                                val indices = predictedIndices ?: return@Button
                                                isRenderingAudio = true
                                                kotlinx.coroutines.CoroutineScope(Dispatchers.IO).launch {
                                                    try {
                                                        val jsonString = context.assets.open("speech_effect_chain_v2.json").bufferedReader().use { it.readText() }
                                                        val inFile = File(filePath)
                                                        val outName = if (inFile.name.startsWith("ENHANCED_")) {
                                                            inFile.name
                                                        } else {
                                                            "ENHANCED_${inFile.name}"
                                                        }
                                                        val outFile = File(inFile.parentFile, outName)

                                                        val py = Python.getInstance()
                                                        val renderer = py.getModule("audio_renderer").callAttr("AudioRenderer")
                                                        val savedPath = renderer.callAttr(
                                                            "render_preset_chain",
                                                            inFile.absolutePath,
                                                            outFile.absolutePath,
                                                            jsonString,
                                                            indices[0],
                                                            indices[1],
                                                            indices[2],
                                                            indices[3],
                                                            indices[4],
                                                            indices[5]
                                                        ).toString()

                                                        kotlinx.coroutines.withContext(Dispatchers.Main) {
                                                            enhancedFilePath = savedPath
                                                            isPlayingEnhanced = true
                                                            isRenderingAudio = false
                                                            exoPlayer.setMediaItem(MediaItem.fromUri(savedPath))
                                                            exoPlayer.prepare()
                                                            exoPlayer.play()
                                                            Toast.makeText(context, "Saved & Playing: $outName", Toast.LENGTH_SHORT).show()
                                                        }
                                                    } catch (e: Exception) {
                                                        Log.e("AUDIO_RENDER", "Failed to render audio", e)
                                                        kotlinx.coroutines.withContext(Dispatchers.Main) {
                                                            isRenderingAudio = false
                                                            Toast.makeText(context, "Render failed: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
                                                        }
                                                    }
                                                }
                                            },
                                            modifier = Modifier.fillMaxWidth(),
                                            enabled = !isRenderingAudio
                                        ) {
                                            Icon(Icons.Default.AutoFixHigh, contentDescription = null)
                                            Spacer(modifier = Modifier.width(8.dp))
                                            Text(
                                                if (isRenderingAudio) "Rendering 6-Stage DSP Chain..."
                                                else if (enhancedFilePath != null) "Re-Render & Play Enhanced Audio (.wav)"
                                                else "Render & Play Enhanced Audio (.wav)"
                                            )
                                        }
                                    }
                                } else {
                                    Button(
                                        onClick = {
                                            isAnalyzing = true
                                            mlErrorMessage = null
                                            kotlinx.coroutines.CoroutineScope(Dispatchers.IO).launch {
                                                try {
                                                    val wavFile = File(filePath)
                                                    val audioDurationSeconds = (wavFile.length() - 44) / 88200.0
                                                    val cnnStartTime = System.currentTimeMillis()

                                                    val py = Python.getInstance()
                                                    val audioProcessor = py.getModule("audio_preprocessor").callAttr("AudioPreprocessor")
                                                    val pythonBytes = audioProcessor.callAttr("process_wav_to_tensor", filePath).toJava(ByteArray::class.java)

                                                    val floatBuffer = ByteBuffer.wrap(pythonBytes).order(ByteOrder.nativeOrder()).asFloatBuffer()
                                                    val spectrogramArray = FloatArray(floatBuffer.remaining())
                                                    floatBuffer.get(spectrogramArray)

                                                    val modelPath = assetFilePath(context, "voxeffects_mobile.ptl")
                                                    val module = LiteModuleLoader.load(modelPath)

                                                    val inputTensor = Tensor.fromBlob(spectrogramArray, longArrayOf(1, 1, 513, 1723))
                                                    val outputTuple = module.forward(IValue.from(inputTensor)).toTuple()

                                                    val denoiseIdx = argmax(outputTuple[0].toTensor().dataAsFloatArray)
                                                    val compIdx = argmax(outputTuple[1].toTensor().dataAsFloatArray)
                                                    val eqIdx = argmax(outputTuple[2].toTensor().dataAsFloatArray)
                                                    val deesserIdx = argmax(outputTuple[3].toTensor().dataAsFloatArray)
                                                    val reverbIdx = argmax(outputTuple[4].toTensor().dataAsFloatArray)

                                                    // Evaluate 6th stage (Limiter) via O(N) 4x local-oversampling True Peak & P_ceiling
                                                    val limiterIdx = audioProcessor.callAttr("recommend_limiter_idx", filePath, compIdx, eqIdx).toInt()

                                                    val cnnProcessingTime = (System.currentTimeMillis() - cnnStartTime) / 1000.0
                                                    val cnnRtf = cnnProcessingTime / audioDurationSeconds
                                                    Log.d("ThesisBenchmark", "CNN RTF: $cnnRtf (Audio: ${audioDurationSeconds}s, Time: ${cnnProcessingTime}s)")

                                                    val results = mapOf(
                                                        "denoise" to parseEffectPreset(context, "de_noise_presets", denoiseIdx),
                                                        "compressor" to parseEffectPreset(context, "dynamic_presets", compIdx),
                                                        "eq" to parseEffectPreset(context, "eq_presets", eqIdx),
                                                        "deesser" to parseEffectPreset(context, "de_esser_presets", deesserIdx),
                                                        "reverb" to parseEffectPreset(context, "reverb_presets", reverbIdx),
                                                        "limiter" to parseEffectPreset(context, "limiter_presets", limiterIdx)
                                                    )

                                                    kotlinx.coroutines.withContext(Dispatchers.Main) {
                                                        predictedIndices = intArrayOf(denoiseIdx, compIdx, eqIdx, deesserIdx, reverbIdx, limiterIdx)
                                                        effectPresets = results
                                                        isAnalyzing = false
                                                    }
                                                } catch (e: Exception) {
                                                    Log.e("ML_PIPELINE", "Error processing audio", e)
                                                    kotlinx.coroutines.withContext(Dispatchers.Main) {
                                                        mlErrorMessage = e.localizedMessage ?: "Unknown ML Error"
                                                        isAnalyzing = false
                                                    }
                                                }
                                            }
                                        },
                                        modifier = Modifier.fillMaxWidth(),
                                        enabled = !isAnalyzing
                                    ) {
                                        Text(if (isAnalyzing) "Analyzing Spectrogram..." else "Generate Recommended Presets")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PermissionDeniedScreen(type: String, onNavigateBack: () -> Unit, onRequestPermission: () -> Unit) {
    Scaffold(
        topBar = { TopAppBar(title = { Text("Permission Required") }, navigationIcon = { IconButton(onClick = onNavigateBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") } }) }
    ) { paddingValues ->
        Column(modifier = Modifier.fillMaxSize().padding(paddingValues).padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
            Icon(Icons.Default.Warning, contentDescription = null, modifier = Modifier.size(64.dp), tint = MaterialTheme.colorScheme.error)
            Spacer(modifier = Modifier.height(16.dp))
            Text("We need $type permission to use this feature.", textAlign = TextAlign.Center)
            Spacer(modifier = Modifier.height(24.dp))
            Button(onClick = onRequestPermission) { Text("Grant Permission") }
        }
    }
}

fun loadVoskModel(context: android.content.Context, onModelReady: (org.vosk.Model) -> Unit, onError: (Exception) -> Unit) {
    kotlinx.coroutines.CoroutineScope(Dispatchers.IO).launch {
        try {
            val modelDir = File(context.filesDir, "model")
            if (!modelDir.exists() || modelDir.listFiles().isNullOrEmpty()) {
                if (modelDir.exists()) modelDir.deleteRecursively()
                modelDir.mkdirs()
                copyAssetRecursive(context, "model-en", modelDir)
            }
            val model = org.vosk.Model(modelDir.absolutePath)
            kotlinx.coroutines.withContext(Dispatchers.Main) {
                onModelReady(model)
            }
        } catch (e: Exception) {
            Log.e("VOSK", "Failed to load model", e)
            kotlinx.coroutines.withContext(Dispatchers.Main) {
                onError(e)
            }
        }
    }
}

fun copyAssetRecursive(context: android.content.Context, assetPath: String, targetDir: File) {
    val assets = context.assets
    val files = assets.list(assetPath)
    if (files == null || files.isEmpty()) {
        assets.open(assetPath).use { input ->
            FileOutputStream(targetDir).use { output ->
                input.copyTo(output)
            }
        }
    } else {
        if (!targetDir.exists()) targetDir.mkdirs()
        for (file in files) {
            val subAssetPath = "$assetPath/$file"
            val subTargetFile = File(targetDir, file)
            copyAssetRecursive(context, subAssetPath, subTargetFile)
        }
    }
}