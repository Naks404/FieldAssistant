package com.example.myapplication

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

@RunWith(AndroidJUnit4::class)
class PipelineParityTest {

    // Point to the directory we pushed via adb
    private val verificationDir = File("/data/local/tmp/generated")

    @Before
    fun setupChaquopy() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
    }

    @Test
    fun testWavHeaders() {
        val audioDir = File(verificationDir, "audio")
        val wavFiles = audioDir.listFiles { _, name -> name.endsWith(".wav") }

        assertTrue("No WAV files found in ${audioDir.absolutePath}", wavFiles?.isNotEmpty() == true)

        wavFiles?.forEach { file ->
            RandomAccessFile(file, "r").use { raf ->
                raf.seek(24)
                val srBytes = ByteArray(4)
                raf.readFully(srBytes)
                val sampleRate = ByteBuffer.wrap(srBytes).order(ByteOrder.LITTLE_ENDIAN).int
                assertEquals("Sample rate mismatch in ${file.name}", 44100, sampleRate)

                raf.seek(34)
                val bpsBytes = ByteArray(2)
                raf.readFully(bpsBytes)
                val bitsPerSample = ByteBuffer.wrap(bpsBytes).order(ByteOrder.LITTLE_ENDIAN).short
                assertEquals("Bit depth mismatch in ${file.name}", 16.toShort(), bitsPerSample)
            }
        }
    }

    @Test
    fun testAudioTensorParity() {
        val jsonFile = File(verificationDir, "benchmark_results.json")
        assertTrue("Benchmark JSON missing at ${jsonFile.absolutePath}", jsonFile.exists())

        val jsonString = jsonFile.readText()
        val rootObj = JSONObject(jsonString)
        val audioResults = rootObj.getJSONObject("audio")

        val py = Python.getInstance()
        val audioProcessor = py.getModule("audio_preprocessor").callAttr("AudioPreprocessor")

        audioResults.keys().forEach { condition ->
            val metrics = audioResults.getJSONObject(condition)
            val expectedMaxDb = metrics.getDouble("tensor_max_db")
            val expectedMinDb = metrics.getDouble("tensor_min_db")

            val wavFile = File(verificationDir, "audio/flite_$condition.wav")
            assertTrue("Missing file: ${wavFile.name}", wavFile.exists())

            val pythonBytes = audioProcessor.callAttr("process_wav_to_tensor", wavFile.absolutePath).toJava(ByteArray::class.java)

            val floatBuffer = ByteBuffer.wrap(pythonBytes).order(ByteOrder.nativeOrder()).asFloatBuffer()
            val spectrogramArray = FloatArray(floatBuffer.remaining())
            floatBuffer.get(spectrogramArray)

            val actualMaxDb = spectrogramArray.maxOrNull() ?: 0f
            val actualMinDb = spectrogramArray.minOrNull() ?: 0f

            assertEquals("Max dB mismatch for $condition", expectedMaxDb, actualMaxDb.toDouble(), 0.1)
            assertEquals("Min dB mismatch for $condition", expectedMinDb, actualMinDb.toDouble(), 0.1)
        }
    }
}