package com.gub.data.database

import com.gub.data.database.daoImpl.SignalDaoImpl
import com.gub.data.database.daoImpl.TrafficStatsDaoImpl
import com.gub.data.database.entity.SignalData
import com.gub.data.database.entity.SignalData.SignalState
import com.gub.data.database.entity.TrafficStats
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.mapdb.DB
import org.mapdb.DBMaker
import java.io.File
import java.time.Instant
import java.util.*
import kotlin.random.Random

object DatabaseFactory {

    val dbFile = File("database/traffic.db")

    init {
        if (dbFile.exists().not())
            dbFile.parentFile.mkdirs()

        CoroutineScope(Dispatchers.IO).launch {
            delay(1000L)
//            DummyData.roadIds
        }
    }

    val db: DB by lazy {
        DBMaker.fileDB(dbFile)
            .fileMmapEnable()
            .transactionEnable()
            .make()
    }
}

object DummyData {

    val signalDao = SignalDaoImpl()
    val trafficStatsDao = TrafficStatsDaoImpl()

    val roadIds = listOf("road_north", "road_south", "road_east", "road_west")

    fun insertDummySignalData(count: Int = 100) {
        val now = Instant.now()
        repeat(count) { i ->
            val randomRoad = roadIds.random()
            val randomOffset = Random.nextLong(0, 3600L * 4) // last 4 hours
            val state = SignalState.values().random()

            val signal = SignalData(
                id = UUID.randomUUID().toString(),
                timestamp = now.minusSeconds(randomOffset),
                roadId = randomRoad,
                signalState = state
            )
            signalDao.insert(signal)
        }
        println("✅ Inserted $count dummy signal entries.")
    }

    fun insertDummyTrafficStatsData(count: Int = 100) {
        val now = Instant.now()
        repeat(count) { i ->
            val randomRoad = roadIds.random()
            val randomOffset = Random.nextLong(0, 3600L * 2) // last 2 hours
            val duration = Random.nextLong(10, 60) // measurement duration in seconds
            val vehicles = Random.nextInt(3, 20)   // vehicle count

            val stats = TrafficStats(
                id = UUID.randomUUID().toString(),
                timestamp = now.minusSeconds(randomOffset),
                measureDuration = duration,
                roadId = randomRoad,
                vehicleCount = vehicles
            )
            trafficStatsDao.insert(stats)
        }
        println("✅ Inserted $count dummy traffic stats entries.")
    }

    init {
        insertDummySignalData(100)
        insertDummyTrafficStatsData(100)
    }
}