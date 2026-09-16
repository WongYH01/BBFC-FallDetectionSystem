package com.bbfc.notification.out.telegram;

import com.bbfc.notification.config.TelegramConfig;
import com.bbfc.notification.core.port.NotificationChannel;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

@Component
public class TelegramBotClient implements NotificationChannel {

    private final RestClient restClient;
    private final String chatGroupId;

    public TelegramBotClient(TelegramConfig telegramConfig,  RestClient.Builder restClientBuilder){
        this.chatGroupId = telegramConfig.chatGroupId();
        this.restClient = restClientBuilder
                .baseUrl(telegramConfig.baseUrl() + "/bot" + telegramConfig.botToken())
                .build();
    }

    private record SendMessageRequest(
            @JsonProperty("chat_id") String chatId,
            String text,
            @JsonProperty("parse_mode") String parseMode
    ){ }


    @Override
    public void sendAlert(String message){
        restClient.post()
                .uri("/sendMessage")
                .body(new SendMessageRequest(chatGroupId, message, "HTML"))
                .retrieve()
                .toBodilessEntity();
    }
}
