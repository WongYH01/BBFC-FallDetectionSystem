package com.bbfc.notification.testsupport;

import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.testcontainers.service.connection.ServiceConnection;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

/**
 * Base for tests that boot the full context.
 *
 * <p>These overrides live here rather than in a properties file on purpose. Real environment
 * variables and the imported .env both outrank config files, so once a developer exports
 * TELEGRAM_BOT_TOKEN to run the app, a file-based override would silently lose and the suite
 * would start talking to the real bot. Inlined @SpringBootTest properties outrank both.
 * NotificationApplicationTests.doesNotLoadRealCredentials is the tripwire.
 */
@SpringBootTest(properties = {
        "telegram.bot-token=test-token",
        "telegram.chat-group-id=-1000000000000",
        // Closed port: anything that tries to call out fails fast instead of hanging.
        "telegram.base-url=http://127.0.0.1:1",
        "telegram.message-template=Fall in {roomName} at {time}, confidence {confidence} (Ref {eventId})",
        // Both background triggers off — tests drive the scheduler and the handler directly.
        "telegram.polling-enabled=false",
        "alerting.escalation.scheduler-enabled=false"
})
@Testcontainers
public abstract class AbstractIntegrationTest {

    @Container
    @ServiceConnection
    static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:15");
}
