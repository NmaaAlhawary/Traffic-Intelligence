package com.gub.data.database.entity

import kotlinx.serialization.Contextual
import kotlinx.serialization.Serializable
import java.time.Instant
import java.util.*

data class TrafficStats(
    val id: String = UUID.randomUUID().toString(),
    val timestamp: Instant = Instant.now(),
    val measureDuration: Long,
    val roadId: String,
    val vehicleCount: Int
) : java.io.Serializable
