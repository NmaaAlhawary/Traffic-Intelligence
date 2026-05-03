package com.gub.data.database.daoImpl

import com.gub.data.database.DatabaseFactory
import com.gub.data.database.dao.TrafficStatsDao
import com.gub.data.database.entity.TrafficStats
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId

class TrafficStatsDaoImpl : TrafficStatsDao {

    private val map = DatabaseFactory.db
        .hashMap("traffic_stats", org.mapdb.Serializer.STRING, org.mapdb.Serializer.JAVA)
        .createOrOpen() as MutableMap<String, TrafficStats>

    override fun insert(data: TrafficStats) {
        map[data.id] = data
        DatabaseFactory.db.commit()
    }

    override fun getById(id: String): TrafficStats? = map[id]

    override fun getAll(): List<TrafficStats> = map.values.toList()

    override fun getByHour(hour: Int): List<TrafficStats> {
        return map.values.filter {
            val time = it.timestamp.atZone(ZoneId.systemDefault()).toLocalTime()
            time.hour == hour
        }
    }

    override fun getByDate(date: String): List<TrafficStats> {
        val targetDate = LocalDate.parse(date)
        return map.values.filter {
            val localDate = it.timestamp.atZone(ZoneId.systemDefault()).toLocalDate()
            localDate == targetDate
        }
    }

    override fun getRecentWithin(seconds: Long): List<TrafficStats> {
        val cutoff = Instant.now().minusSeconds(seconds)
        return map.values.filter { it.timestamp.isAfter(cutoff) }
    }

    override fun delete(id: String) {
        map.remove(id)
        DatabaseFactory.db.commit()
    }
}