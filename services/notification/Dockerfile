# ---------- build stage ----------
FROM maven:3.9.6-eclipse-temurin-21 AS build
WORKDIR /app

# Pre-fetch deps (faster rebuilds)
COPY pom.xml .
RUN mvn -q -B dependency:go-offline

# Build the jar
COPY src ./src
RUN mvn -q -DskipTests clean package

# ---------- extract layers ----------
# Splits the fat jar into dependencies / loader / application, so the run
# stage's COPY layers below only invalidate for the layer that actually changed.
FROM eclipse-temurin:21-jre-jammy AS extract
WORKDIR /app
COPY --from=build /app/target/notification-0.0.1-SNAPSHOT.jar application.jar
RUN java -Djarmode=tools -jar application.jar extract --layers --destination extracted

# ---------- run stage ----------
FROM eclipse-temurin:21-jre-jammy
WORKDIR /app

# (optional) run as non-root
RUN useradd -r -s /usr/sbin/nologin appuser
USER appuser

# Optional JVM opts (tune as you like)
ENV JAVA_OPTS=""

EXPOSE 8081

COPY --from=extract /app/extracted/dependencies/ ./
COPY --from=extract /app/extracted/spring-boot-loader/ ./
COPY --from=extract /app/extracted/snapshot-dependencies/ ./
COPY --from=extract /app/extracted/application/ ./

# (optional) simple TCP healthcheck without curl/wget
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD bash -lc '</dev/tcp/127.0.0.1/8081' || exit 1

# Run it
ENTRYPOINT ["bash","-lc","java $JAVA_OPTS -jar application.jar"]
