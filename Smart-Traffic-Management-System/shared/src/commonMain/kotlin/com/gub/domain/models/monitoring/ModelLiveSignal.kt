package com.gub.domain.models.monitoring

import kotlinx.serialization.Serializable

@Serializable
data class ModelLiveSignal(
    val roadName: String,
    val east: Road,
    val west: Road,
    val north: Road,
    val south: Road,
    val waitingTime: Long
) {

    @Serializable
    data class Road(
        val name: String,
        val type: SignalType,
        val vehicleCount: Int,
    ) {
        enum class SignalType {
            RED,
            YELLOW,
            GREEN
        }
    }
}