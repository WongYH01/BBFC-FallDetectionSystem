package com.bbfc.notification.in.rest;

import com.bbfc.notification.core.domain.ClipAlreadyAttachedException;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.port.ClipStorageException;
import com.bbfc.notification.core.service.AlertNotFoundException;
import com.bbfc.notification.core.service.AttachClipService;
import com.bbfc.notification.core.service.ClipTooLargeException;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.http.HttpMethod;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

import java.nio.charset.StandardCharsets;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.BDDMockito.given;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(ClipController.class)
class ClipControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private AttachClipService attachClipService;

    private static MockMultipartFile clipPart() {
        return new MockMultipartFile("file", "pose_skeleton.mp4", "video/mp4",
                "fake-mp4-bytes".getBytes(StandardCharsets.UTF_8));
    }

    @Test
    void storedClipReturns202() throws Exception {
        given(attachClipService.attach(any(), any())).willReturn(true);

        mockMvc.perform(multipart(HttpMethod.PUT, "/events/{id}/clip", "FE-1").file(clipPart()))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.eventId").value("FE-1"))
                .andExpect(jsonPath("$.status").value("stored"))
                .andExpect(jsonPath("$.sentToTelegram").value(true));
    }

    @Test
    void clipStoredButNotDeliveredStillReturns202() throws Exception {
        given(attachClipService.attach(any(), any())).willReturn(false);

        mockMvc.perform(multipart(HttpMethod.PUT, "/events/{id}/clip", "FE-1").file(clipPart()))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.sentToTelegram").value(false));
    }

    @Test
    void unknownEventReturns404() throws Exception {
        given(attachClipService.attach(any(), any()))
                .willThrow(new AlertNotFoundException(new EventId("FE-1")));

        mockMvc.perform(multipart(HttpMethod.PUT, "/events/{id}/clip", "FE-1").file(clipPart()))
                .andExpect(status().isNotFound());
    }

    @Test
    void secondClipReturns409() throws Exception {
        given(attachClipService.attach(any(), any()))
                .willThrow(new ClipAlreadyAttachedException(new EventId("FE-1")));

        mockMvc.perform(multipart(HttpMethod.PUT, "/events/{id}/clip", "FE-1").file(clipPart()))
                .andExpect(status().isConflict());
    }

    @Test
    void oversizeClipReturns413() throws Exception {
        given(attachClipService.attach(any(), any()))
                .willThrow(new ClipTooLargeException(9_000_000L, 8_388_608L));

        mockMvc.perform(multipart(HttpMethod.PUT, "/events/{id}/clip", "FE-1").file(clipPart()))
                .andExpect(status().isPayloadTooLarge());
    }

    @Test
    void servletLevelUploadLimitReturns413() throws Exception {
        given(attachClipService.attach(any(), any()))
                .willThrow(new MaxUploadSizeExceededException(8_388_608L));

        mockMvc.perform(multipart(HttpMethod.PUT, "/events/{id}/clip", "FE-1").file(clipPart()))
                .andExpect(status().isPayloadTooLarge());
    }

    @Test
    void storageFailureReturns500() throws Exception {
        given(attachClipService.attach(any(), any()))
                .willThrow(new ClipStorageException("storage down"));

        mockMvc.perform(multipart(HttpMethod.PUT, "/events/{id}/clip", "FE-1").file(clipPart()))
                .andExpect(status().isInternalServerError());
    }

    @Test
    void missingFilePartReturns400() throws Exception {
        mockMvc.perform(multipart(HttpMethod.PUT, "/events/{id}/clip", "FE-1"))
                .andExpect(status().isBadRequest());
    }
}
