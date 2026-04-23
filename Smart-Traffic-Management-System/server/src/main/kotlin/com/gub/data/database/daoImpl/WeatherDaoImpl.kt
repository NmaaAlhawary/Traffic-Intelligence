package com.gub.data.database.daoImpl

import com.gub.data.database.DatabaseFactory
import com.gub.data.database.dao.WeatherDao
import com.gub.data.database.entity.WeatherData
import java.time.Instant

class WeatherDaoImpl : WeatherDao {

    private val map = DatabaseFactory.db
        .hashMap("weather", org.mapdb.Serializer.STRING, org.mapdb.Serializer.JAVA)
        .createOrOpen() as MutableMap<String, WeatherData>

    override fun insert(data: WeatherData) {
        map[data.id] = data
        DatabaseFactory.db.commit()
    }

    override fun getAll(): List<WeatherData> = map.values.toList()

    override fun getRecent(hours: Long): List<WeatherData> {
        val cutoff = Instant.now().minusSeconds(hours * 3600)
        return map.values.filter { it.timestamp.isAfter(cutoff) }
    }
}