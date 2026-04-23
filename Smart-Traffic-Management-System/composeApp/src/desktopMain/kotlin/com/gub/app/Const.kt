package com.gub.app

object Const {

    const val BASE_URL = "localhost"
//    const val BASE_URL = "34.87.172.238"

    private const val DEFAULT_SERVER_URL = BASE_URL

    private const val DEFAULT_STREAM_HOST = BASE_URL

    private const val SERVER_PORT = 8080

    private const val STREAM_PORT = 1234



    const val SERVER_URL = "$DEFAULT_SERVER_URL:$SERVER_PORT"

    const val STREAM_URL = "http://$DEFAULT_STREAM_HOST:$STREAM_PORT"

    const val STREAM_WEBSOCKET_URL = "ws://$DEFAULT_STREAM_HOST:$STREAM_PORT/ws"
}