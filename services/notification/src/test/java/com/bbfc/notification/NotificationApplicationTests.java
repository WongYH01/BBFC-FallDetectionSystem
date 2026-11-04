package com.bbfc.notification;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;

import com.bbfc.notification.config.TelegramConfig;
import com.bbfc.notification.testsupport.AbstractIntegrationTest;

class NotificationApplicationTests extends AbstractIntegrationTest {

	@Autowired
	private TelegramConfig telegramConfig;

	@Test
	void contextLoads() {
	}

	@Autowired
	private Environment environment;

	@Test
	void openInViewStaysOff() {
		assertThat(environment.getProperty("spring.jpa.open-in-view"))
				.as("with open-in-view on, a locked re-read returns the request's cached entity and its save "
						+ "can undo a concurrent dispatch; see application.properties")
				.isEqualTo("false");
	}

	// Compared as a boolean so a failure never prints the real token into the surefire report.
	@Test
	void doesNotLoadRealCredentials() {
		assertThat(telegramConfig.botToken().equals("test-token"))
				.as("test context must not pick up .env or a real TELEGRAM_BOT_TOKEN")
				.isTrue();
	}
}
