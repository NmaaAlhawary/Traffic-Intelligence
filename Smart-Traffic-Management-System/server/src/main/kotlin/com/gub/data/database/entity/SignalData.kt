package com.gub.data.database.entity

import kotlinx.serialization.Serializable
import kotlinx.serialization.Contextual
import java.time.Instant
import java.util.*

data class SignalData(
    val id: String = UUID.randomUUID().toString(),
    val timestamp: Instant = Instant.now(),
    val roadId: String,
    val signalState: SignalState
) : java.io.Serializable {
    @Serializable
    enum class SignalState {
        RED, GREEN, YELLOW
    }
}