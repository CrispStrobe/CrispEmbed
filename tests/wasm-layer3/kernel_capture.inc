/* TEST OVERLAY ONLY. Insert after the actual conversion barrier in mul_mat.
 * Writes the actual selected-plan workspace, not a type-trait surrogate. */
    if (ith == 0 && getenv("CRISPEMBED_LAYER3_KERNEL") &&
            strncmp(src0->name, "blk.3.", 6) == 0) {
        const char * directory = getenv("CRISPEMBED_LAYER3_KERNEL");
        GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
        GGML_ASSERT(src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1));
        GGML_ASSERT(ne12 == 1 && ne13 == 1 && ne11 == 12);
        GGML_ASSERT(vec_dot_type == GGML_TYPE_Q8_0 || vec_dot_type == GGML_TYPE_Q8_K);
        GGML_ASSERT(src1->type != vec_dot_type);
        const size_t converted_bytes = ggml_row_size(vec_dot_type, ne10)*ne11;
        const void * converted = params->wdata;
        char file[2048];
#define L3_WRITE(suffix, pointer, size) do { \
        GGML_ASSERT(snprintf(file, sizeof(file), "%s/%s.%s", directory, src0->name, suffix) < (int) sizeof(file)); \
        FILE * stream = fopen(file, "wb"); GGML_ASSERT(stream); \
        GGML_ASSERT(fwrite(pointer, 1, size, stream) == size); GGML_ASSERT(fclose(stream) == 0); \
    } while (0)
        L3_WRITE("input.f32", src1->data, (size_t) ne10*ne11*sizeof(float));
        L3_WRITE("converted.bin", converted, converted_bytes);
        L3_WRITE("weight.bin", src0->data, ggml_nbytes(src0));
        float dots[12];
        for (int row = 0; row < 12; ++row) {
            type_traits_cpu[src0->type].vec_dot((int) ne10, &dots[row], 0,
                src0->data, 0, (const char *) converted + row*ggml_row_size(vec_dot_type, ne10), 0, 1);
        }
        L3_WRITE("dots.f32", dots, sizeof(dots));
        GGML_ASSERT(snprintf(file, sizeof(file), "%s/%s.json", directory, src0->name) < (int) sizeof(file));
        FILE * stream = fopen(file, "wb"); GGML_ASSERT(stream);
        fprintf(stream, "{\"weight\":\"%s\",\"weight_type\":%d,\"vec_dot_type\":%d,"
            "\"input_shape\":[%lld,%lld],\"output_shape\":[%lld,%lld],\"weight_bytes\":%zu,"
            "\"converted_bytes\":%zu,\"wasm_round_plan\":%d,\"use_ref\":%d,"
            "\"provenance\":\"actual mul_mat selected-plan workspace after conversion barrier; dot is selected trait first-channel evaluation\"}\n",
            src0->name, src0->type, vec_dot_type, (long long)ne10, (long long)ne11,
            (long long)ne01, (long long)ne11, ggml_nbytes(src0), converted_bytes,
            params->wasm_q8_0_round, params->use_ref);
        GGML_ASSERT(fclose(stream) == 0);
#undef L3_WRITE
    }
