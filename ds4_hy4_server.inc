static void hy4_render_schemas(buf *out, const char *schemas) {
    buf_puts(out,
        "# Tools\n\nYou may call one or more functions to assist with the user query."
        "\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
    const char *p = schemas;
    while (*p) {
        char *raw = NULL;
        json_ws(&p);
        if (!json_raw_value(&p, &raw)) break;
        bool emitted = false;
        if (!append_glm_tool_schema_json(out, raw, &emitted)) {
            buf_puts(out, raw);
            emitted = true;
        }
        if (emitted) buf_putc(out, '\n');
        free(raw);
    }
    buf_puts(out,
        "</tools>\n\nFor function call returns, you should first print " HY4_TOOLS_OPEN
        "\nFor each function call, you should return object like:\n"
        HY4_TOOL_OPEN "{function-name}" HY4_KEY_OPEN "{arg-key-1}" HY4_KEY_CLOSE
        HY4_VALUE_OPEN "{arg-value-1}" HY4_VALUE_CLOSE HY4_KEY_OPEN "{arg-key-2}" HY4_KEY_CLOSE
        HY4_VALUE_OPEN "{arg-value-2}" HY4_VALUE_CLOSE "..." HY4_TOOL_CLOSE
        "\nAt the end of function call returns, you should print " HY4_TOOLS_CLOSE);
}

static void hy4_render_calls(buf *out, const tool_calls *calls) {
    if (calls->len == 0) return;
    if (calls->raw_tool_text && !strncmp(calls->raw_tool_text, HY4_TOOLS_OPEN, sizeof(HY4_TOOLS_OPEN)-1)) {
        buf_puts(out, calls->raw_tool_text);
        return;
    }
    buf_puts(out, HY4_TOOLS_OPEN);
    for (int i = 0; i < calls->len; i++) {
        const tool_call *call = &calls->v[i];
        buf_puts(out, HY4_TOOL_OPEN);
        buf_puts(out, call->name);
        json_args args = {0};
        if (json_args_parse(call->arguments, &args)) {
            for (int j = 0; j < args.len; j++) {
                const json_arg *arg = &args.v[j];
                buf_puts(out, HY4_KEY_OPEN);
                append_glm_tag_body_text(out, arg->key, HY4_KEY_CLOSE);
                buf_puts(out, HY4_KEY_CLOSE HY4_VALUE_OPEN);
                append_glm_tag_body_text(out, arg->value, HY4_VALUE_CLOSE);
                buf_puts(out, HY4_VALUE_CLOSE);
            }
        }
        json_args_free(&args);
        buf_puts(out, HY4_TOOL_CLOSE);
    }
    buf_puts(out, HY4_TOOLS_CLOSE);
}

static char *render_hy4_chat_prompt(const chat_msgs *msgs, const char *schemas, ds4_think_mode mode) {
    buf out = {0};
    int last_user = -1;
    for (int i = 0; i < msgs->len; i++)
        if (!strcmp(msgs->v[i].role, "user")) last_user = i;
    const bool think = ds4_think_mode_enabled(mode);
    const bool leading_system = msgs->len && (!strcmp(msgs->v[0].role, "system") ||
                                               !strcmp(msgs->v[0].role, "developer"));
    if (!leading_system) {
        buf_puts(&out, HY4_CHAT_START "system" HY4_CHAT_MIDDLE);
        if (schemas && *schemas) hy4_render_schemas(&out, schemas);
        buf_puts(&out, HY4_CHAT_EFFORT);
        buf_puts(&out, ds4_hy4_reasoning_effort(mode));
        buf_puts(&out, HY4_CHAT_END);
    }
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *msg = &msgs->v[i];
        const char *role = !strcmp(msg->role, "developer") ? "system" : msg->role;
        const bool assistant = !strcmp(role, "assistant");
        const bool tool = !strcmp(role, "tool") || !strcmp(role, "function");
        buf_puts(&out, HY4_CHAT_START);
        buf_puts(&out, tool ? "tool" : role);
        buf_puts(&out, HY4_CHAT_MIDDLE);
        if (i == 0 && leading_system && schemas && *schemas) {
            hy4_render_schemas(&out, schemas);
            if (msg->content && msg->content[0]) buf_puts(&out, "\n\n");
        }
        if (assistant) {
            buf_puts(&out, HY4_THINK_OPEN);
            if (think && i > last_user && msg->reasoning) buf_puts(&out, msg->reasoning);
            buf_puts(&out, HY4_THINK_CLOSE);
        }
        if (tool) {
            buf_puts(&out, HY4_RESULT_OPEN);
            append_glm_tag_body_text(&out, msg->content ? msg->content : "", HY4_RESULT_CLOSE);
            buf_puts(&out, HY4_RESULT_CLOSE);
        } else if (msg->content) {
            buf_puts(&out, msg->content);
        }
        if (assistant) hy4_render_calls(&out, &msg->calls);
        if (i == 0 && leading_system) {
            buf_puts(&out, HY4_CHAT_EFFORT);
            buf_puts(&out, ds4_hy4_reasoning_effort(mode));
        }
        buf_puts(&out, HY4_CHAT_END);
    }
    if (!msgs->len || strcmp(msgs->v[msgs->len-1].role, "assistant")) {
        buf_puts(&out, HY4_CHAT_START "assistant" HY4_CHAT_MIDDLE HY4_THINK_OPEN);
        if (!think) buf_puts(&out, HY4_THINK_CLOSE);
    }
    return buf_take(&out);
}
