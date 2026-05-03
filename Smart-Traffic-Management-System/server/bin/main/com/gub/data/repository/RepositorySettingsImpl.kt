package com.gub.data.repository

import com.gub.data.database.dao.SignalDao
import com.gub.data.database.dao.TrafficStatsDao
import com.gub.data.database.dao.WeatherDao
import com.gub.data.database.entity.SignalData
import com.gub.data.database.entity.TrafficStats
import com.gub.data.database.entity.WeatherData
import com.gub.data.models.JsonExportWrapper
import com.gub.domain.repository.RepositorySettings
import com.gub.domain.model.ExportJson
import com.gub.domain.model.ExportSignalData
import com.gub.domain.model.ExportTrafficStats
import com.gub.domain.model.ExportWeatherData
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileInputStream
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class RepositorySettingsImpl(
    private val signalDao: SignalDao,
    private val weatherDao: WeatherDao,
    private val trafficStatsDao: TrafficStatsDao
) : RepositorySettings {

    private val dateFormatter = DateTimeFormatter.ISO_INSTANT

    override suspend fun importCsv(content: ByteArray) {
        val csv = content.toString(Charsets.UTF_8)
        val sections = csv.split("weather_data.csv")

        val trafficLines = sections[0]
            .substringAfter("traffic_stats.csv")
            .trim()
            .lines()
            .drop(1) // skip header

        val signalLines = sections[1]
            .substringBefore("id,timestamp,temperature,humidity,windSpeed,visibility")
            .trim()
            .lines()
            .drop(1)

        val weatherLines = sections[1]
            .substringAfter("id,timestamp,temperature,humidity,windSpeed,visibility")
            .trim()
            .lines()
            .drop(1)

        val trafficStats = trafficLines.mapNotNull { line ->
            val parts = line.split(",")
            if (parts.size == 5) {
                TrafficStats(
                    id = parts[0],
                    timestamp = Instant.parse(parts[1]),
                    measureDuration = parts[2].toLongOrNull() ?: 0,
                    roadId = parts[3],
                    vehicleCount = parts[4].toIntOrNull() ?: 0
                )
            } else null
        }

        val signalData = signalLines.mapNotNull { line ->
            val parts = line.split(",")
            if (parts.size == 4) {
                SignalData(
                    id = parts[0],
                    timestamp = Instant.parse(parts[1]),
                    roadId = parts[2],
                    signalState = SignalData.SignalState.valueOf(parts[3])
                )
            } else null
        }

        val weatherData = weatherLines.mapNotNull { line ->
            val parts = line.split(",")
            if (parts.size == 6) {
                WeatherData(
                    id = parts[0],
                    timestamp = Instant.parse(parts[1]),
                    temperature = parts[2].toDoubleOrNull() ?: 0.0,
                    humidity = parts[3].toDoubleOrNull() ?: 0.0,
                    windSpeed = parts[4].toDoubleOrNull() ?: 0.0,
                    visibility = parts[5].toDoubleOrNull() ?: 0.0
                )
            } else null
        }

        // Save to database
        trafficStats.forEach { trafficStatsDao.insert(it) }
        signalData.forEach { signalDao.insert(it) }
        weatherData.forEach { weatherDao.insert(it) }
    }

    override suspend fun importJson(content: ByteArray) {
        val json = Json { ignoreUnknownKeys = true }
        val parsed = json.decodeFromString<JsonExportWrapper>(content.toString(Charsets.UTF_8))

        parsed.trafficStats.forEach { trafficStatsDao.insert(it) }
        parsed.signalData.forEach { signalDao.insert(it) }
        parsed.weatherData.forEach { weatherDao.insert(it) }
    }

    override suspend fun exportCsv(): String {
        val exportDir = File("exports/csv")
        exportDir.mkdirs()

        val trafficStats = trafficStatsDao.getAll()
        val signalData = signalDao.getAll()
        val weatherData = weatherDao.getAll()

        // Write CSV files
        val files = listOf(
            File(exportDir, "traffic_stats.csv") to listOf("id,timestamp,measureDuration,roadId,vehicleCount") +
                    trafficStats.map {
                        "${it.id},${it.timestamp.formatTimestamp()},${it.measureDuration},${it.roadId},${it.vehicleCount}"
                    },
            File(exportDir, "signal_data.csv") to listOf("id,timestamp,roadId,signalState") +
                    signalData.map {
                        "${it.id},${it.timestamp.formatTimestamp()},${it.roadId},${it.signalState}"
                    },
            File(exportDir, "weather_data.csv") to listOf("id,timestamp,temperature,humidity,windSpeed,visibility") +
                    weatherData.map {
                        "${it.id},${it.timestamp.formatTimestamp()},${it.temperature},${it.humidity},${it.windSpeed},${it.visibility}"
                    }
        )

        files.forEach { (file, lines) ->
            file.printWriter().use { out -> lines.forEach { out.println(it) } }
        }

        // Create ZIP file
        val zipFile = File("exports/traffic_export.zip")
        ZipOutputStream(zipFile.outputStream()).use { zipOut ->
            files.forEach { (file, _) ->
                FileInputStream(file).use { input ->
                    val entry = ZipEntry(file.name)
                    zipOut.putNextEntry(entry)
                    input.copyTo(zipOut)
                }
            }
        }

        return zipFile.absolutePath
    }

    override suspend fun exportJson(): String {
        val outputDir = File("exports/json")
        outputDir.mkdirs()

        val trafficStats = trafficStatsDao.getAll()
        val signalData = signalDao.getAll()
        val weatherData = weatherDao.getAll()

        val jsonData = ExportJson(
            exportedAt = Instant.now().formatTimestamp(),
            trafficStats = trafficStats.map {
                ExportTrafficStats(
                    id = it.id,
                    timestamp = it.timestamp.formatTimestamp(),
                    measureDuration = it.measureDuration,
                    roadId = it.roadId,
                    vehicleCount = it.vehicleCount
                )
            },
            signalData = signalData.map {
                ExportSignalData(
                    id = it.id,
                    timestamp = it.timestamp.formatTimestamp(),
                    roadId = it.roadId,
                    signalState = it.signalState.name
                )
            },
            weatherData = weatherData.map {
                ExportWeatherData(
                    id = it.id,
                    timestamp = it.timestamp.formatTimestamp(),
                    temperature = it.temperature,
                    humidity = it.humidity,
                    windSpeed = it.windSpeed,
                    visibility = it.visibility
                )
            }
        )

        val json = Json { prettyPrint = true }.encodeToString(jsonData)
        val file = File(outputDir, "traffic_data.json")
        file.writeText(json)

        return file.absolutePath
    }

    private fun Instant.formatTimestamp(): String {
        val formatter = DateTimeFormatter.ofPattern("dd MMMM yyyy hh:mm a")
            .withLocale(Locale.ENGLISH)
            .withZone(ZoneId.of("GMT+6"))

        return formatter.format(this)
    }
}