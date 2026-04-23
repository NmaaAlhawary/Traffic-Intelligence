package com.gub.data.service.dashboard

import com.gub.data.database.dao.SignalDao
import com.gub.data.database.dao.TrafficStatsDao
import com.gub.data.database.entity.SignalData
import com.gub.data.service.common.WeatherService
import com.gub.domain.models.dashboard.ModelSystemOverview
import com.sun.management.OperatingSystemMXBean
import java.io.File
import java.lang.management.ManagementFactory
import java.time.Duration
import java.time.Instant

class SystemOverviewService(
    private val signalDao: SignalDao,
    private val trafficStatsDao: TrafficStatsDao,
    private val weatherService: WeatherService,
) {

    suspend fun systemOverview(): ModelSystemOverview {

        return ModelSystemOverview(
            systemHealth = calculateSystemHealth(),
            aiResponseTime = getLastAIResponseTime(),
            avgWaitTime = getAverageVehicleWaitTime(),
            currentFlow = getCurrentTrafficFlowRate(),
            weather = weatherService.getCurrentWeather()
        )
    }

    @Suppress("DefaultLocale")
    private fun calculateSystemHealth(): Double {
        // Memory usage
        val runtime = Runtime.getRuntime()
        val totalMem = runtime.totalMemory().toDouble()
        val freeMem = runtime.freeMemory().toDouble()
        val usedMemRatio = (totalMem - freeMem) / totalMem

        // Real-time CPU load (not load average)
        val osBean = ManagementFactory.getPlatformMXBean(OperatingSystemMXBean::class.java)
        val cpuLoad = osBean.systemCpuLoad.coerceIn(0.0, 1.0)

        // Disk usage
        val root = File("/")
        val totalDisk = root.totalSpace.toDouble()
        val freeDisk = root.freeSpace.toDouble()
        val diskUsageRatio = (totalDisk - freeDisk) / totalDisk

        // Weighted average (you can tweak the weights based on your use case)
        val memoryWeight = 0.4
        val cpuWeight = 0.4
        val diskWeight = 0.2

        val pressureScore = (usedMemRatio * memoryWeight) +
                (cpuLoad * cpuWeight) +
                (diskUsageRatio * diskWeight)

        val healthScore = ((1.0 - pressureScore) * 100).coerceIn(0.0, 100.0)
        return String.format("%.2f", healthScore).toDouble()  // Round to 2 decimal places
    }

    // TODO: Implement real AI response time tracking
    private fun getLastAIResponseTime(): Double {
        return (80..150).random().toDouble()
    }

    // Calculates average wait time during RED signals (in seconds)
    fun getAverageVehicleWaitTime(): Double {
        val redSignals = signalDao.getAll().filter { it.signalState == SignalData.SignalState.RED }

        if (redSignals.isEmpty()) return 0.0

        // Sort by timestamp for accurate duration calculation
        val sorted = redSignals.sortedBy { it.timestamp }

        var totalWaitTimeSeconds = 0L
        var redCount = 0

        for (i in 1 until sorted.size) {
            val prev = sorted[i - 1]
            val curr = sorted[i]

            if (curr.roadId == prev.roadId) {
                val duration = Duration.between(prev.timestamp, curr.timestamp).seconds
                totalWaitTimeSeconds += duration
                redCount++
            }
        }

        return if (redCount == 0) 0.0 else (totalWaitTimeSeconds.toDouble() / redCount) / 60
    }

    // Calculates recent vehicle flow rate (vehicles per minute)
    fun getCurrentTrafficFlowRate(): Double {
        val now = Instant.now()
        val recentStats = trafficStatsDao.getAll()

        if (recentStats.isEmpty()) return 0.0

        val totalVehicles = recentStats.sumOf { it.vehicleCount }
        val earliestTimestamp = recentStats.minOf { it.timestamp }
        val durationSeconds = Duration.between(earliestTimestamp, now).seconds.coerceAtLeast(1)

        val flowRatePerMinute = (totalVehicles.toDouble() / durationSeconds) * 60.0
        return flowRatePerMinute
    }
}