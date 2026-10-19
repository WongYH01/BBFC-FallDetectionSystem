package com.bbfc.notification.in.rest;

import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.KeypointsAlreadyAttachedException;
import com.bbfc.notification.core.port.ClipContent;
import com.bbfc.notification.core.service.AttachKeypointsService;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.http.HttpMethod;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import java.nio.charset.StandardCharsets;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.BDDMockito.given;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(KeypointsController.class)
class KeypointsControllerTest {

    private static final byte[] NPZ = "PK\u0003\u0004rest-of-zip".getBytes(StandardCharsets.ISO_8859_1);
    private static final byte[] PARQUET = "PAR1rest-of-parquetPAR1".getBytes(StandardCharsets.US_ASCII);

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private AttachKeypointsService attachKeypointsService;

    private static MockMultipartFile part(String filename, byte[] bytes) {
        return new MockMultipartFile("file", filename, "application/octet-stream", bytes);
    }

    private org.springframework.test.web.servlet.ResultActions upload(MockMultipartFile file) throws Exception {
        return mockMvc.perform(multipart(HttpMethod.PUT, "/events/{id}/keypoints", "FE-1").file(file));
    }

    @Test
    void npzIsStoredUnderTheEventNameAndReturns202() throws Exception {
        upload(part("../whatever.NPZ", NPZ))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.eventId").value("FE-1"))
                .andExpect(jsonPath("$.status").value("stored"));

        ArgumentCaptor<ClipContent> content = ArgumentCaptor.forClass(ClipContent.class);
        verify(attachKeypointsService).attach(any(EventId.class), content.capture());
        assertThat(content.getValue().filename()).isEqualTo("FE-1.npz");
        assertThat(content.getValue().contentType()).isEqualTo("application/octet-stream");
    }

    @Test
    void parquetIsStoredWithItsMediaType() throws Exception {
        upload(part("fall.parquet", PARQUET)).andExpect(status().isAccepted());

        ArgumentCaptor<ClipContent> content = ArgumentCaptor.forClass(ClipContent.class);
        verify(attachKeypointsService).attach(any(EventId.class), content.capture());
        assertThat(content.getValue().filename()).isEqualTo("FE-1.parquet");
        assertThat(content.getValue().contentType()).isEqualTo("application/vnd.apache.parquet");
    }

    @Test
    void unsupportedExtensionReturns415() throws Exception {
        upload(part("fall.csv", NPZ)).andExpect(status().isUnsupportedMediaType());

        verify(attachKeypointsService, never()).attach(any(), any());
    }

    @Test
    void wrongMagicBytesReturn415() throws Exception {
        upload(part("fall.npz", PARQUET)).andExpect(status().isUnsupportedMediaType());

        verify(attachKeypointsService, never()).attach(any(), any());
    }

    @Test
    void secondUploadReturns409() throws Exception {
        given(attachKeypointsService.attach(any(), any()))
                .willThrow(new KeypointsAlreadyAttachedException(new EventId("FE-1")));

        upload(part("fall.npz", NPZ)).andExpect(status().isConflict());
    }
}
