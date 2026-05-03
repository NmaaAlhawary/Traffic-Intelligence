package com.gub.data.database.daoImpl

import com.gub.data.database.DatabaseFactory
import com.gub.data.database.dao.SignalDao
import com.gub.data.database.entity.SignalData
import java.time.ZoneOffset

class SignalDaoImpl : SignalDao {

    private val map = DatabaseFactory.db
        .hashMap(
            "signal_data",
            org.mapdb.Serializer.STRING,
            org.mapdb.Serializer.JAVA
        ).createOrOpen() as MutableMap<String, SignalData>

    override fun insert(data: SignalData) {
        map[data.id] = data
        DatabaseFactory.db.commit()
    }

    override fun getByRouteId(id: String): List<SignalData> {
        return map.values.filter { it.roadId == id }
    }

    override fun getAll(): List<SignalData> {
        return map.values.toList()
    }

    override fun getByHour(hour: Int): List<SignalData> {
        return map.values.filter {
            val dataHour = it.timestamp.atZone(ZoneOffset.UTC).hour
            dataHour == hour
        }
    }

    override fun delete(id: String) {
        map.remove(id)
        DatabaseFactory.db.commit()
    }
}