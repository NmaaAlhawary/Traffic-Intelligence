package com.gub.data.service.dashboard

import com.gub.data.database.dao.SignalDao
import com.gub.data.database.dao.TrafficStatsDao
import com.gub.domain.models.dashboard.ModelLiveTraffic
import kotlin.math.abs

class ServiceTrafficMeasure(
    private val signalDao: SignalDao,
    private val trafficStatsDao: TrafficStatsDao
) {

    companion object {
        const val ROAD_CAPACITY_PER_MIN = 60 // Adjust this based on real road capacity
    }

    fun getLiveTrafficSummary(): ModelLiveTraffic {
        val allStats = (0 until 24)
            .map { hour -> hour to getStatsForHour(hour) }
            .filter { (_, stats) -> stats.vehicleCount > 0 }

        if (allStats.size < 2) {
            return ModelLiveTraffic(
                vehicle = ModelLiveTraffic.Vehicle(0, 0.0, false),
                congestion = ModelLiveTraffic.Congestion(0.0, 0.0, false)
            )
        }

        val (_, stats2) = allStats[allStats.lastIndex - 1]
        val (_, stats1) = allStats.last()

        val vehicleFlow1 = stats1.vehiclePerMinute
        val vehicleFlow2 = stats2.vehiclePerMinute
        val vehicleDiff = vehicleFlow1 - vehicleFlow2

        val congestionDiff = stats1.congestionIndex - stats2.congestionIndex

        return ModelLiveTraffic(
            vehicle = ModelLiveTraffic.Vehicle(
                count = stats1.vehicleCount,
                difference = abs(vehicleDiff),
                upWards = vehicleDiff >= 0
            ),
            congestion = ModelLiveTraffic.Congestion(
                count = stats1.congestionIndex * 100,
                difference = abs(congestionDiff * 100),
                upWards = congestionDiff >= 0
            )
        )
    }

    private data class HourStats(
        val vehicleCount: Int,
        val vehiclePerMinute: Double,
        val congestionIndex: Double
    )

    private fun getStatsForHour(hour: Int): HourStats {
        val stats = trafficStatsDao.getByHour(hour)

        val vehicleCount = stats.sumOf { it.vehicleCount }
        val totalDuration = stats.sumOf { it.measureDuration }.coerceAtLeast(1)

        val vehiclePerMinute = (vehicleCount.toDouble() / totalDuration) * 60.0
        val congestionIndex = vehiclePerMinute / ROAD_CAPACITY_PER_MIN

        return HourStats(vehicleCount, vehiclePerMinute, congestionIndex)
    }
}