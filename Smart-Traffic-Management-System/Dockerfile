# --- Stage 1: Build ---
FROM gradle:8.2-jdk17 AS builder

WORKDIR /app

# ✅ Copy everything from project root (includes gradlew, shared, server)
COPY . .

# ✅ Give exec permission just in case
RUN chmod +x ./gradlew

# ✅ Build only server module with shared dependencies
RUN ./gradlew :server:installDist --no-daemon

# --- Stage 2: Run ---
FROM eclipse-temurin:17-jre

WORKDIR /app

# ✅ Copy the server output only
COPY --from=builder /app/server/build/install/server /app

EXPOSE 8080

CMD ["./bin/server"]