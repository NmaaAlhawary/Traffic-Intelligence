package com.gub.data.database.dao

import com.gub.data.database.entity.TrafficStats


interface TrafficStatsDao {

    fun insert(data: TrafficStats)

    fun getById(id: String): TrafficStats?

    fun getAll(): List<TrafficStats>

    fun getByHour(hour: Int): List<TrafficStats>

    fun getByDate(date: String): List<TrafficStats> // e.g., "2025-07-11"

    fun getRecentWithin(seconds: Long): List<TrafficStats>

    fun delete(id: String)
}