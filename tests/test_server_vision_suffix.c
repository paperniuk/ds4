/* Exercise the server's real multimodal suffix paths without loading a model. */
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#define ds4_engine_is_glm_dsa test_engine_is_glm_dsa
#define ds4_engine_is_qwen4 test_engine_is_qwen4
#define ds4_tokenize_rendered_chat test_tokenize_rendered_chat
#include "../ds4_server.c"
#undef ds4_engine_is_glm_dsa
#undef ds4_engine_is_qwen4
#undef ds4_tokenize_rendered_chat

#include <assert.h>

typedef struct {
    server_model_syntax syntax;
} test_engine;

bool test_engine_is_glm_dsa(ds4_engine *engine) {
    return ((test_engine *)engine)->syntax == SERVER_MODEL_SYNTAX_GLM;
}

bool test_engine_is_qwen4(ds4_engine *engine) {
    return ((test_engine *)engine)->syntax == SERVER_MODEL_SYNTAX_QWEN;
}

void test_tokenize_rendered_chat(ds4_engine *engine, const char *text,
                                 ds4_tokens *out) {
    (void)engine;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++)
        ds4_tokens_push(out, *p);
}

static void append_image(ds4_tokens *prompt, ds4_vision_span *image,
                         bool wrapped, int fingerprint) {
    if (wrapped) ds4_tokens_push(prompt, 248053);
    image->token_start = (uint32_t)prompt->len;
    image->embedding.token_count = 2;
    image->embedding.fingerprint[0] = (uint8_t)fingerprint;
    ds4_tokens_push(prompt, 248056);
    ds4_tokens_push(prompt, 248056);
    if (wrapped) ds4_tokens_push(prompt, 248054);
}

static void assert_same_tokens(const ds4_tokens *got, const ds4_tokens *want) {
    assert(got->len == want->len);
    assert(memcmp(got->v, want->v, (size_t)want->len * sizeof(*want->v)) == 0);
}

static void test_suffix(server_model_syntax syntax, bool wrapped) {
    test_engine engine = {.syntax = syntax};
    server s = {.engine = (ds4_engine *)&engine};
    char markers[2][SERVER_IMAGE_MARKER_BYTES] = {"[image-one]", "[image-two]"};
    ds4_vision_span images[2] = {0};
    request req = {.model_syntax = syntax, .images = images,
                   .image_markers = markers, .image_count = 2,
                   .prompt_text = "AB[image-one]CD[image-two]EF"};
    ds4_tokens_push(&req.prompt, 'A');
    ds4_tokens_push(&req.prompt, 'B');
    append_image(&req.prompt, &images[0], wrapped, 1);
    ds4_tokens_push(&req.prompt, 'C');
    ds4_tokens_push(&req.prompt, 'D');
    const int second_image_boundary = req.prompt.len;
    append_image(&req.prompt, &images[1], wrapped, 2);
    ds4_tokens_push(&req.prompt, 'E');
    ds4_tokens_push(&req.prompt, 'F');
    const uint32_t first_start = images[0].token_start;
    const uint32_t second_start = images[1].token_start;

    /* Append two images to a text-only live checkpoint. */
    server_slot slot = {.session = ds4_session_new_test_checkpoint(req.prompt.v, 2)};
    ds4_tokens actual = {0};
    const char *suffix = req.prompt_text + 2;
    assert(build_live_prompt_suffix(&s, &slot, &req, suffix, &actual));
    assert_same_tokens(&actual, &req.prompt);
    assert(images[0].token_start == first_start && images[1].token_start == second_start);
    ds4_tokens_free(&actual);
    ds4_session_free_test_checkpoint(slot.session);

    /* A live checkpoint already contains the first image. */
    slot.session = ds4_session_new_test_checkpoint(req.prompt.v, second_image_boundary);
    ds4_session_set_test_images(slot.session, images, 1);
    suffix = strstr(req.prompt_text, markers[1]);
    assert(build_live_prompt_suffix(&s, &slot, &req, suffix, &actual));
    assert_same_tokens(&actual, &req.prompt);
    assert(images[0].token_start == first_start && images[1].token_start == second_start);
    ds4_tokens_free(&actual);
    ds4_session_free_test_checkpoint(slot.session);

    /* A retained live prefix can contain tokens absent from the visible replay. */
    ds4_tokens live = {0};
    for (int i = 0; i < second_image_boundary; i++) ds4_tokens_push(&live, req.prompt.v[i]);
    ds4_tokens_push(&live, 901);
    ds4_tokens_push(&live, 902);
    slot.session = ds4_session_new_test_checkpoint(live.v, live.len);
    ds4_session_set_test_images(slot.session, images, 1);
    assert(build_live_prompt_suffix(&s, &slot, &req, suffix, &actual));
    for (int i = second_image_boundary; i < req.prompt.len; i++)
        ds4_tokens_push(&live, req.prompt.v[i]);
    assert_same_tokens(&actual, &live);
    assert(images[0].token_start == first_start && images[1].token_start == second_start + 2);
    ds4_tokens_free(&actual);
    ds4_tokens_free(&live);
    ds4_session_free_test_checkpoint(slot.session);

    /* Invalid image slices must be rejected. */
    images[0].token_start = wrapped ? 0 : (uint32_t)req.prompt.len;
    slot.session = ds4_session_new_test_checkpoint(req.prompt.v, 2);
    suffix = req.prompt_text + 2;
    assert(!build_live_prompt_suffix(&s, &slot, &req, suffix, &actual));
    ds4_tokens_free(&actual);
    ds4_session_free_test_checkpoint(slot.session);
    ds4_tokens_free(&req.prompt);
}

int main(void) {
    test_suffix(SERVER_MODEL_SYNTAX_QWEN, true);
    test_suffix(SERVER_MODEL_SYNTAX_GLM, true);
    test_suffix(SERVER_MODEL_SYNTAX_DEEPSEEK, false);
    test_suffix(SERVER_MODEL_SYNTAX_DEEPSEEK41, false);
    puts("server multimodal suffix: PASS");
    return 0;
}
