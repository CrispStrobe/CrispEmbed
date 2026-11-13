"""Generate a test-only prefill/image observer from the immutable historical driver."""
import hashlib
from pathlib import Path

SOURCE = 'a6164205620c71427a39c97ab386591406c4ca73'
DRIVER_SHA = 'e8c8363d183465600d22ccc2419bd18103a0ca3596fdf9abf5077b2f802dc634'
CMAKE_SHA = 'e226d2764eb16e1904cd59bbda72064f0ff4e8e0e0629dc6913fdfb3db563b89'
TARGET = 'lighton4b-book-image-layer0-probe'
PROBE = r'''
    // Test executable only: repeat layer0 with the actual planner/cache boundaries.
    // Close every tile's ordinary output and final state before interpreting a capture.
    if (!model.text.decoder.upstream_physical || ids.size() != 266 || mods.size() != 266 ||
        !lighton4b_source_prefill_chunks_mode(true) || initial.hidden.size() != 32 ||
        initial.hidden[0].size() != size_t(266)*2560 || initial.cache.layers.size() != 32) return 30;
    std::vector<lighton4b_prefill_chunk> image_plan;
    if (!lighton4b_source_prefill_plan(mods,image_plan) || image_plan.size()<2 ||
        image_plan[0].offset!=0 || image_plan[0].tokens!=4 || image_plan[1].offset!=4 || image_plan[1].tokens!=16 ||
        !std::all_of(mods.begin(),mods.begin()+4,[](int x){return x==0;}) ||
        !std::all_of(mods.begin()+4,mods.begin()+256,[](int x){return x==1;}) ||
        !std::all_of(mods.begin()+256,mods.end(),[](int x){return x==0;})) return 30;
    auto bits=[](const std::vector<float>&a,const std::vector<float>&b){
        return a.size()==b.size() && (a.empty() || !std::memcmp(a.data(),b.data(),a.size()*sizeof(float)));
    };
    auto cache_equal=[&](const lighton4b_linear_cache&a,const lighton4b_linear_cache&b){
        return a.tokens==b.tokens && a.bf16_activations==b.bf16_activations &&
          a.upstream_physical==b.upstream_physical && a.physical_native_ops==b.physical_native_ops &&
          a.hf_tiled_out==b.hf_tiled_out && a.physical_q4_repack==b.physical_q4_repack &&
          a.physical_gdn_source_ops==b.physical_gdn_source_ops && a.physical_q6_tiled==b.physical_q6_tiled &&
          a.physical_source_prefill_chunks==b.physical_source_prefill_chunks && bits(a.conv,b.conv) && bits(a.recurrent,b.recurrent);
    };
    lighton4b_linear_cache replay_cache;
    bool captured_image=false;
    int replay_offset=0;
    for (const auto&part:image_plan){
        if(part.offset!=replay_offset || replay_cache.tokens!=replay_offset) return 31;
        const auto preserved=replay_cache;
        lighton4b_linear_block_outputs observed;
        if(!lighton4b_linear_block_cpu(model.text.decoder.layers[0],prefill.hidden.data()+size_t(part.offset)*2560,
           part.tokens,threads,part.offset?&replay_cache:nullptr,observed) ||
           !cache_equal(preserved,replay_cache) || observed.cache.tokens!=part.offset+part.tokens ||
           observed.hidden.size()!=size_t(part.tokens)*2560 ||
           std::memcmp(observed.hidden.data(),initial.hidden[0].data()+size_t(part.offset)*2560,observed.hidden.size()*sizeof(float))) return 31;
        if(part.offset==4){
            if(captured_image || replay_cache.tokens!=4 || replay_cache.conv.size()!=32768 || replay_cache.recurrent.size()!=524288) return 32;
            emit("image.layer0.input",std::vector<float>(prefill.hidden.begin()+size_t(part.offset)*2560,prefill.hidden.begin()+size_t(part.offset+part.tokens)*2560));
            emit("image.layer0.input_norm",observed.input_norm);
            emit("image.layer0.attention",observed.attention);
            emit("image.layer0.attention_residual",observed.attention_residual);
            emit("image.layer0.post_norm",observed.post_norm);
            emit("image.layer0.gate",observed.gate);
            emit("image.layer0.up",observed.up);
            emit("image.layer0.activation",observed.activation);
            emit("image.layer0.down",observed.down);
            emit("image.layer0.hidden",observed.hidden);
            emit("image.layer0.cache_before.conv",preserved.conv);
            emit("image.layer0.cache_before.recurrent",preserved.recurrent);
            emit("image.layer0.cache_after.conv",observed.cache.conv);
            emit("image.layer0.cache_after.recurrent",observed.cache.recurrent);
            projected_linear_outputs projected;
            if(!linear_grouped_cpu(model.text.decoder.layers[0].linear,16,observed.input_norm.data(),1,part.tokens,threads,
               preserved.conv.data(),preserved.recurrent.data(),projected,model.text.decoder.physical_weights_cache.get()) ||
               !bits(projected.attention,observed.attention) || !bits(projected.core.state,observed.cache.recurrent) ||
               !bits(projected.core.conv_history,observed.cache.conv)) return 32;
            emit("image.layer0.qkv",projected.qkv);emit("image.layer0.z",projected.z);
            emit("image.layer0.a",projected.a);emit("image.layer0.b",projected.b);
            emit("image.layer0.core_values",projected.core.values);
            captured_image=true;
        }
        replay_cache=std::move(observed.cache);replay_offset+=part.tokens;
    }
    if(!captured_image || replay_offset!=266 || !cache_equal(replay_cache,initial.cache.layers[0].linear)) return 33;
    std::ofstream(out+"/image-layer0-capture.json") << "{\"schema\":\"lighton4b-book-image-layer0-capture/v1\",\"tokens\":16,\"offset\":4,\"cache_before_tokens\":4,\"cache_after_tokens\":20,\"all_layer0_tiles_hidden_exact\":true,\"final_layer0_cache_exact\":true,\"previous_caches_immutable\":true,\"core_replay_attn_state_conv_exact\":true,\"decoded_quality_qualified\":false}\n";
'''
STOP = r'''
    // Only this separately built test executable stops here. Ordinary driver is untouched.
    std::cout << "BOOK_IMAGE_LAYER0_CAPTURE PASS actual_prefill=1 decoded=0" << std::endl;
    return 0;
'''


def generate(driver):
    if hashlib.sha256(driver).hexdigest() != DRIVER_SHA:
        raise ValueError('historical native driver drift')
    text = driver.decode()
    anchor = '    std::vector<float> first;\n'
    stop = '    emit("prefill.logits", first);\n'
    if text.count(anchor) != 1 or text.count(stop) != 1:
        raise ValueError('closed driver insertion anchors')
    return text.replace(anchor, PROBE + '\n' + anchor).replace(stop, stop + STOP)


def prepare(root):
    root = Path(root)
    cmake = root / 'CMakeLists.txt'
    if hashlib.sha256(cmake.read_bytes()).hexdigest() != CMAKE_SHA:
        raise ValueError('historical CMake drift')
    generated = generate((root / 'tests/lighton4b_native_document.cpp').read_bytes())
    output = root / 'tools/book-image-layer0-probe/generated_driver.cpp'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated)
    cmake.write_text(cmake.read_text() + '\n# Test-only Kaggle image observer; never installed.\n'
        + 'add_executable(' + TARGET + ' EXCLUDE_FROM_ALL tools/book-image-layer0-probe/generated_driver.cpp src/lighton4b_tokenizer.cpp)\n'
        + 'target_link_libraries(' + TARGET + ' PRIVATE crispembed-qwen35-delta crispembed-core)\n')
    return dict(driver_sha256=hashlib.sha256(generated.encode()).hexdigest(), target=TARGET)
