package com.bbfc.notification.in.rest;

import org.springframework.http.HttpStatus;
import org.springframework.http.ProblemDetail;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

import com.bbfc.notification.core.domain.ClipAlreadyAttachedException;
import com.bbfc.notification.core.port.ClipStorageException;
import com.bbfc.notification.core.service.AlertNotFoundException;
import com.bbfc.notification.core.service.ClipTooLargeException;

@RestControllerAdvice
public class ApiExceptionHandler {
    
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ProblemDetail handleValidation(MethodArgumentNotValidException ex) {
        ProblemDetail problem = ProblemDetail.forStatus(HttpStatus.BAD_REQUEST);
        problem.setTitle("Validation failed");
        String detail = ex.getBindingResult().getFieldErrors().stream()
                .map(e -> e.getField() + ": " + e.getDefaultMessage())
                .reduce((a, b) -> a + "; " + b)
                .orElse("Invalid request");
        problem.setDetail(detail);
        return problem;
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ProblemDetail handleIllegalArgument(IllegalArgumentException ex) {
        ProblemDetail problem = ProblemDetail.forStatus(HttpStatus.BAD_REQUEST);
        problem.setTitle("Invalid request");
        problem.setDetail(ex.getMessage());
        return problem;
    }

    @ExceptionHandler(AlertNotFoundException.class)
    public ProblemDetail handleAlertNotFound(AlertNotFoundException ex) {
        return problem(HttpStatus.NOT_FOUND, "Alert not found", ex.getMessage());
    }

    @ExceptionHandler(ClipAlreadyAttachedException.class)
    public ProblemDetail handleClipAlreadyAttached(ClipAlreadyAttachedException ex) {
        return problem(HttpStatus.CONFLICT, "Clip already attached", ex.getMessage());
    }

    @ExceptionHandler(ClipTooLargeException.class)
    public ProblemDetail handleClipTooLarge(ClipTooLargeException ex) {
        return problem(HttpStatus.PAYLOAD_TOO_LARGE, "Clip too large", ex.getMessage());
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ProblemDetail handleUploadTooLarge(MaxUploadSizeExceededException ex) {
        return problem(HttpStatus.PAYLOAD_TOO_LARGE, "Clip too large", ex.getMessage());
    }

    @ExceptionHandler(ClipStorageException.class)
    public ProblemDetail handleClipStorageFailure(ClipStorageException ex) {
        // The caller still holds the bytes, so a 5xx invites the retry we want.
        return problem(HttpStatus.INTERNAL_SERVER_ERROR, "Clip could not be stored", ex.getMessage());
    }

    private static ProblemDetail problem(HttpStatus status, String title, String detail) {
        ProblemDetail problem = ProblemDetail.forStatus(status);
        problem.setTitle(title);
        problem.setDetail(detail);
        return problem;
    }
}
