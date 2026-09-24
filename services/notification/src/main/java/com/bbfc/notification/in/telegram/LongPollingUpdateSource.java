package com.bbfc.notification.in.telegram;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;

import com.bbfc.notification.config.TelegramConfig;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.GetUpdatesResponse;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.Update;

@Component
public class LongPollingUpdateSource implements UpdateSource, DisposableBean {

    private static final Logger log = LoggerFactory.getLogger(LongPollingUpdateSource.class);

    private static final int POLL_TIMEOUT_SECONDS = 25;
    private static final long ERROR_BACKOFF_MILLIS = 5_000;
    private static final long SHUTDOWN_GRACE_MILLIS = 5_000;

    private final RestClient restClient;
    private final CallbackQueryHandler callbackQueryHandler;

    private volatile boolean running;
    private volatile Thread pollingThread;

    /**
     * Confirmed-updates cursor, deliberately in memory only. On restart Telegram redelivers up to
     * 24h of unconfirmed updates; the alert row's state is what makes that safe. This is a
     * de-duplication optimisation, not the correctness mechanism.
     */
    private volatile long nextOffset;

    public LongPollingUpdateSource(
            TelegramConfig telegramConfig,
            RestClient.Builder restClientBuilder,
            CallbackQueryHandler callbackQueryHandler) {
        this.restClient = restClientBuilder
                .baseUrl(telegramConfig.baseUrl() + "/bot" + telegramConfig.botToken())
                .build();
        this.callbackQueryHandler = callbackQueryHandler;
    }

    @Override
    public void start() {
        running = true;
        Thread thread = new Thread(this::run, "telegram-long-poller");
        thread.setDaemon(true);
        pollingThread = thread;
        thread.start();
        log.info("Telegram long polling started");
    }

    private void run() {
        while (running) {
            if (!pollOnce()) {
                sleepQuietly();
            }
        }
    }

    /** One getUpdates round trip. Package-private so a test can drive it without a thread. */
    boolean pollOnce() {
        try {
            GetUpdatesResponse response = restClient.get()
                    .uri(uriBuilder -> uriBuilder.path("/getUpdates")
                            .queryParam("offset", nextOffset)
                            .queryParam("timeout", POLL_TIMEOUT_SECONDS)
                            // Server-side per-bot state, so it has to be sent every call.
                            .queryParam("allowed_updates", "[\"callback_query\"]")
                            .build())
                    .retrieve()
                    .body(GetUpdatesResponse.class);

            if (response == null || response.result() == null) {
                return true;
            }
            response.result().forEach(this::consume);
            return true;
        } catch (RestClientResponseException e) {
            if (e.getStatusCode().value() == HttpStatus.CONFLICT.value()) {
                log.error("Telegram returned 409: another long-poller is using this bot token. "
                        + "Only one getUpdates consumer per token is allowed — check for a second "
                        + "instance of this service or another bot sharing the token.");
            } else {
                log.warn("getUpdates failed: {}", e.getMessage());
            }
            return false;
        } catch (RuntimeException e) {
            if (!running || Thread.currentThread().isInterrupted()) {
                return true;
            }
            log.warn("getUpdates failed", e);
            return false;
        }
    }

    private void consume(Update update) {
        try {
            if (update.callbackQuery() != null) {
                callbackQueryHandler.handle(update.callbackQuery());
            }
        } catch (RuntimeException e) {
            // Advance past it regardless, or one poison update blocks the cursor forever.
            log.error("Failed to handle update {}", update.updateId(), e);
        } finally {
            nextOffset = Math.max(nextOffset, update.updateId() + 1);
        }
    }

    private void sleepQuietly() {
        try {
            Thread.sleep(ERROR_BACKOFF_MILLIS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            running = false;
        }
    }

    @Override
    public void stop() {
        running = false;
        Thread thread = pollingThread;
        if (thread == null) {
            return;
        }
        thread.interrupt();
        try {
            thread.join(SHUTDOWN_GRACE_MILLIS);
            if (thread.isAlive()) {
                log.warn("Long poller did not stop within {}ms", SHUTDOWN_GRACE_MILLIS);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        pollingThread = null;
        log.info("Telegram long polling stopped");
    }

    @Override
    public void destroy() {
        stop();
    }

    long nextOffset() {
        return nextOffset;
    }
}
