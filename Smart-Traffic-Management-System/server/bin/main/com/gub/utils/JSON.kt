package com.gub.utils

import kotlinx.serialization.json.Json
import kotlinx.serialization.modules.SerializersModule
import java.time.Instant

val json = Json {
    prettyPrint = true
    encodeDefaults = true
    ignoreUnknownKeys = true
    serializersModule = SerializersModule {
        contextual(Instant::class, InstantSerializer)
    }
}