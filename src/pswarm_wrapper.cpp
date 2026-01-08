#include <napi.h>
#include "ParticleSlam.h"
#include <memory>
#include <vector>

using namespace pswarm;

// ============================================================================
// Helper Functions
// ============================================================================

Vec3 object_to_vec3(const Napi::Object& obj) {
    Vec3 v;
    v.x = obj.Get("x").As<Napi::Number>().FloatValue();
    v.y = obj.Get("y").As<Napi::Number>().FloatValue();
    v.z = obj.Get("z").As<Napi::Number>().FloatValue();
    return v;
}

Napi::Object vec3_to_object(Napi::Env env, const Vec3& v) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("x", Napi::Number::New(env, v.x));
    obj.Set("y", Napi::Number::New(env, v.y));
    obj.Set("z", Napi::Number::New(env, v.z));
    return obj;
}

Napi::Object particle_to_object(Napi::Env env, const Particle& p) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("state", vec3_to_object(env, p.state));
    obj.Set("timestamp", Napi::Number::New(env, p.timestamp));
    return obj;
}

Mat3 object_to_mat3(const Napi::Object& obj) {
    Mat3 m;
    Napi::Array arr = obj.Get("m").As<Napi::Array>();
    for (int i = 0; i < 9; i++) {
        m.m[i] = arr.Get(i).As<Napi::Number>().FloatValue();
    }
    return m;
}

Napi::Object mat3_to_object(Napi::Env env, const Mat3& m) {
    Napi::Object obj = Napi::Object::New(env);
    Napi::Array arr = Napi::Array::New(env, 9);
    for (int i = 0; i < 9; i++) {
        arr.Set(i, Napi::Number::New(env, m.m[i]));
    }
    obj.Set("m", arr);
    return obj;
}

Chunk object_to_chunk(const Napi::Object& obj) {
    Chunk chunk;
    chunk.timestamp = obj.Get("timestamp").As<Napi::Number>().DoubleValue();
    
    Napi::Array cells = obj.Get("cells").As<Napi::Array>();
    for (int i = 0; i < 60; i++) {
        Napi::Array row = cells.Get(i).As<Napi::Array>();
        for (int j = 0; j < 60; j++) {
            Napi::Object cell = row.Get(j).As<Napi::Object>();
            chunk.cells[i][j].num_pos = cell.Get("num_pos").As<Napi::Number>().Uint32Value();
            chunk.cells[i][j].num_neg = cell.Get("num_neg").As<Napi::Number>().Uint32Value();
        }
    }
    
    return chunk;
}

Napi::Object chunk_to_object(Napi::Env env, const Chunk& chunk) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("timestamp", Napi::Number::New(env, chunk.timestamp));
    
    Napi::Array cells = Napi::Array::New(env, 60);
    for (int i = 0; i < 60; i++) {
        Napi::Array row = Napi::Array::New(env, 60);
        for (int j = 0; j < 60; j++) {
            Napi::Object cell = Napi::Object::New(env);
            cell.Set("num_pos", Napi::Number::New(env, chunk.cells[i][j].num_pos));
            cell.Set("num_neg", Napi::Number::New(env, chunk.cells[i][j].num_neg));
            row.Set(j, cell);
        }
        cells.Set(i, row);
    }
    obj.Set("cells", cells);
    
    return obj;
}

Map* object_to_map(const Napi::Object& obj) {
    Map* map = new Map();
    
    map->width = obj.Get("width").As<Napi::Number>().Int32Value();
    map->height = obj.Get("height").As<Napi::Number>().Int32Value();
    map->min_x = obj.Get("min_x").As<Napi::Number>().FloatValue();
    map->min_y = obj.Get("min_y").As<Napi::Number>().FloatValue();
    map->max_x = obj.Get("max_x").As<Napi::Number>().FloatValue();
    map->max_y = obj.Get("max_y").As<Napi::Number>().FloatValue();
    map->cell_size = obj.Get("cell_size").As<Napi::Number>().FloatValue();
    
    size_t num_cells = static_cast<size_t>(map->width) * static_cast<size_t>(map->height);
    map->cells = new ChunkCell[num_cells];
    
    Napi::Array cells = obj.Get("cells").As<Napi::Array>();
    for (size_t i = 0; i < num_cells; i++) {
        Napi::Object cell = cells.Get(i).As<Napi::Object>();
        map->cells[i].num_pos = cell.Get("num_pos").As<Napi::Number>().Uint32Value();
        map->cells[i].num_neg = cell.Get("num_neg").As<Napi::Number>().Uint32Value();
    }
    
    return map;
}

Napi::Object map_to_object(Napi::Env env, const Map* map) {
    Napi::Object obj = Napi::Object::New(env);
    
    obj.Set("width", Napi::Number::New(env, map->width));
    obj.Set("height", Napi::Number::New(env, map->height));
    obj.Set("min_x", Napi::Number::New(env, map->min_x));
    obj.Set("min_y", Napi::Number::New(env, map->min_y));
    obj.Set("max_x", Napi::Number::New(env, map->max_x));
    obj.Set("max_y", Napi::Number::New(env, map->max_y));
    obj.Set("cell_size", Napi::Number::New(env, map->cell_size));
    
    size_t num_cells = static_cast<size_t>(map->width) * static_cast<size_t>(map->height);
    Napi::Array cells = Napi::Array::New(env, num_cells);
    
    for (size_t i = 0; i < num_cells; i++) {
        Napi::Object cell = Napi::Object::New(env);
        cell.Set("num_pos", Napi::Number::New(env, map->cells[i].num_pos));
        cell.Set("num_neg", Napi::Number::New(env, map->cells[i].num_neg));
        cells.Set(i, cell);
    }
    obj.Set("cells", cells);
    
    return obj;
}

Napi::Object measurement_to_object(Napi::Env env, const Measurement& m) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("covariance", mat3_to_object(env, m.covariance));
    obj.Set("mean", vec3_to_object(env, m.mean));
    obj.Set("is_gaussian", Napi::Boolean::New(env, m.is_gaussian));
    return obj;
}

// ============================================================================
// ParticleSlam Wrapper Class
// ============================================================================

class ParticleSlamWrapper : public Napi::ObjectWrap<ParticleSlamWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports) {
        Napi::Function func = DefineClass(env, "ParticleSlam", {
            InstanceMethod("init", &ParticleSlamWrapper::Init),
            InstanceMethod("apply_step", &ParticleSlamWrapper::ApplyStep),
            InstanceMethod("ingest_visual_measurement", &ParticleSlamWrapper::IngestVisualMeasurement),
            InstanceMethod("evaluate_and_resample", &ParticleSlamWrapper::EvaluateAndResample),
            InstanceMethod("download_chunk_states", &ParticleSlamWrapper::DownloadChunkStates),
            InstanceMethod("download_chunk_states_for_particle", &ParticleSlamWrapper::DownloadChunkStatesForParticle),
            InstanceMethod("download_scores", &ParticleSlamWrapper::DownloadScores),
            InstanceMethod("download_current_particle_states", &ParticleSlamWrapper::DownloadCurrentParticleStates),
            InstanceMethod("get_current_chunk_count", &ParticleSlamWrapper::GetCurrentChunkCount),
            InstanceMethod("get_current_timestep", &ParticleSlamWrapper::GetCurrentTimestep),
            InstanceMethod("get_num_particles", &ParticleSlamWrapper::GetNumParticles),
            InstanceMethod("bake_best_particle_map", &ParticleSlamWrapper::BakeBestParticleMap),
            InstanceMethod("accumulate_map_from_trajectories", &ParticleSlamWrapper::AccumulateMapFromTrajectories),
            InstanceMethod("accumulate_map_from_map", &ParticleSlamWrapper::AccumulateMapFromMap),
            InstanceMethod("set_global_map", &ParticleSlamWrapper::SetGlobalMap),
            InstanceMethod("uniform_initialize_particles", &ParticleSlamWrapper::UniformInitializeParticles),
            InstanceMethod("prune_particles_outside_map", &ParticleSlamWrapper::PruneParticlesOutsideMap),
            InstanceMethod("bake_global_map_best_particle", &ParticleSlamWrapper::BakeGlobalMapBestParticle),
            InstanceMethod("calculate_measurement", &ParticleSlamWrapper::CalculateMeasurement)
        });

        Napi::FunctionReference* constructor = new Napi::FunctionReference();
        *constructor = Napi::Persistent(func);
        env.SetInstanceData(constructor);

        exports.Set("ParticleSlam", func);
        return exports;
    }

    ParticleSlamWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<ParticleSlamWrapper>(info) {
        Napi::Env env = info.Env();

        if (info.Length() < 3) {
            Napi::TypeError::New(env, "Expected at least 3 arguments").ThrowAsJavaScriptException();
            return;
        }

        int num_particles = info[0].As<Napi::Number>().Int32Value();
        int max_trajectory_length = info[1].As<Napi::Number>().Int32Value();
        int max_chunk_length = info[2].As<Napi::Number>().Int32Value();

        float cell_size_m = 0.1f;
        float pos_weight = 0.7f;
        float neg_weight = 0.4f;
        float alpha_prior = 1.0f;
        float beta_prior = 1.5f;
        uint8_t measurement_saturation = 200;

        if (info.Length() > 3) cell_size_m = info[3].As<Napi::Number>().FloatValue();
        if (info.Length() > 4) pos_weight = info[4].As<Napi::Number>().FloatValue();
        if (info.Length() > 5) neg_weight = info[5].As<Napi::Number>().FloatValue();
        if (info.Length() > 6) alpha_prior = info[6].As<Napi::Number>().FloatValue();
        if (info.Length() > 7) beta_prior = info[7].As<Napi::Number>().FloatValue();
        if (info.Length() > 8) measurement_saturation = info[8].As<Napi::Number>().Uint32Value();

        particle_slam_ = std::make_unique<ParticleSlam>(
            num_particles, max_trajectory_length, max_chunk_length,
            cell_size_m, pos_weight, neg_weight, alpha_prior, beta_prior, measurement_saturation
        );
    }

private:
    std::unique_ptr<ParticleSlam> particle_slam_;

    Napi::Value Init(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        unsigned long long random_seed = 1234ULL;
        if (info.Length() > 0) {
            random_seed = info[0].As<Napi::Number>().Int64Value();
        }
        
        particle_slam_->init(random_seed);
        return env.Undefined();
    }

    Napi::Value ApplyStep(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        if (info.Length() < 2) {
            Napi::TypeError::New(env, "Expected at least 2 arguments").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        Vec3 dx_step = object_to_vec3(info[0].As<Napi::Object>());
        double timestamp = info[1].As<Napi::Number>().DoubleValue();
        
        float pos_std = 1.6e-3f;
        float yaw_std = 1e-3f;
        
        if (info.Length() > 2) pos_std = info[2].As<Napi::Number>().FloatValue();
        if (info.Length() > 3) yaw_std = info[3].As<Napi::Number>().FloatValue();
        
        particle_slam_->apply_step(dx_step, timestamp, pos_std, yaw_std);
        return env.Undefined();
    }

    Napi::Value IngestVisualMeasurement(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        if (info.Length() < 1) {
            Napi::TypeError::New(env, "Expected 1 argument").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        Chunk chunk = object_to_chunk(info[0].As<Napi::Object>());
        int result = particle_slam_->ingest_visual_measurement(chunk);
        
        return Napi::Number::New(env, result);
    }

    Napi::Value EvaluateAndResample(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        if (info.Length() < 1) {
            Napi::TypeError::New(env, "Expected 1 argument").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        int chunk_index = info[0].As<Napi::Number>().Int32Value();
        particle_slam_->evaluate_and_resample(chunk_index);
        
        return env.Undefined();
    }

    Napi::Value DownloadChunkStates(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        if (info.Length() < 1) {
            Napi::TypeError::New(env, "Expected 1 argument").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        int max_chunks = info[0].As<Napi::Number>().Int32Value();
        int num_particles = particle_slam_->get_num_particles();
        
        std::vector<Vec3> h_chunk_states(max_chunks * num_particles);
        particle_slam_->download_chunk_states(h_chunk_states.data(), max_chunks);
        
        Napi::Array result = Napi::Array::New(env, max_chunks);
        for (int i = 0; i < max_chunks; i++) {
            Napi::Array particles = Napi::Array::New(env, num_particles);
            for (int j = 0; j < num_particles; j++) {
                particles.Set(j, vec3_to_object(env, h_chunk_states[i * num_particles + j]));
            }
            result.Set(i, particles);
        }
        
        return result;
    }

    Napi::Value DownloadChunkStatesForParticle(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        if (info.Length() < 2) {
            Napi::TypeError::New(env, "Expected 2 arguments").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        int particle_idx = info[0].As<Napi::Number>().Int32Value();
        int max_chunks = info[1].As<Napi::Number>().Int32Value();
        
        std::vector<Vec3> h_chunk_states(max_chunks);
        particle_slam_->download_chunk_states_for_particle(h_chunk_states.data(), particle_idx, max_chunks);
        
        Napi::Array result = Napi::Array::New(env, max_chunks);
        for (int i = 0; i < max_chunks; i++) {
            result.Set(i, vec3_to_object(env, h_chunk_states[i]));
        }
        
        return result;
    }

    Napi::Value DownloadScores(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        int num_particles = particle_slam_->get_num_particles();
        std::vector<float> h_scores(num_particles);
        particle_slam_->download_scores(h_scores.data());
        
        Napi::Array result = Napi::Array::New(env, num_particles);
        for (int i = 0; i < num_particles; i++) {
            result.Set(i, Napi::Number::New(env, h_scores[i]));
        }
        
        return result;
    }

    Napi::Value DownloadCurrentParticleStates(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        int num_particles = particle_slam_->get_num_particles();
        std::vector<Particle> h_current_states(num_particles);
        particle_slam_->download_current_particle_states(h_current_states.data());
        
        Napi::Array result = Napi::Array::New(env, num_particles);
        for (int i = 0; i < num_particles; i++) {
            result.Set(i, particle_to_object(env, h_current_states[i]));
        }
        
        return result;
    }

    Napi::Value GetCurrentChunkCount(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        return Napi::Number::New(env, particle_slam_->get_current_chunk_count());
    }

    Napi::Value GetCurrentTimestep(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        return Napi::Number::New(env, particle_slam_->get_current_timestep());
    }

    Napi::Value GetNumParticles(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        return Napi::Number::New(env, particle_slam_->get_num_particles());
    }

    Napi::Value BakeBestParticleMap(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        int start_chunk_index = 0;
        if (info.Length() > 0 && info[0].IsNumber()) {
            start_chunk_index = info[0].As<Napi::Number>().Int32Value();
        }
        
        Map* map = particle_slam_->bake_best_particle_map(start_chunk_index);
        Napi::Object result = map_to_object(env, map);
        delete map;
        
        return result;
    }

    Napi::Value AccumulateMapFromTrajectories(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        if (info.Length() < 1) {
            Napi::TypeError::New(env, "Expected 1 argument").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        int chunk_index = info[0].As<Napi::Number>().Int32Value();
        particle_slam_->accumulate_map_from_trajectories(chunk_index);
        
        return env.Undefined();
    }

    Napi::Value AccumulateMapFromMap(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        if (info.Length() < 1) {
            Napi::TypeError::New(env, "Expected 1 argument").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        int chunk_index = info[0].As<Napi::Number>().Int32Value();
        particle_slam_->accumulate_map_from_map(chunk_index);
        
        return env.Undefined();
    }

    Napi::Value SetGlobalMap(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        if (info.Length() < 1) {
            Napi::TypeError::New(env, "Expected 1 argument").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        Map* map = object_to_map(info[0].As<Napi::Object>());
        particle_slam_->set_global_map(*map);
        delete map;
        
        return env.Undefined();
    }

    Napi::Value UniformInitializeParticles(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        particle_slam_->uniform_initialize_particles();
        return env.Undefined();
    }

    Napi::Value PruneParticlesOutsideMap(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        particle_slam_->prune_particles_outside_map();
        return env.Undefined();
    }

    Napi::Value BakeGlobalMapBestParticle(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        int start_chunk_index = 0;
        if (info.Length() > 0 && info[0].IsNumber()) {
            start_chunk_index = info[0].As<Napi::Number>().Int32Value();
        }
        
        Map* map = particle_slam_->bake_global_map_best_particle(start_chunk_index);
        Napi::Object result = map_to_object(env, map);
        delete map;
        
        return result;
    }

    Napi::Value CalculateMeasurement(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        
        Measurement m = particle_slam_->calculate_measurement();
        return measurement_to_object(env, m);
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

Napi::Value LoadMapFromFile(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Expected 1 argument").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string filename = info[0].As<Napi::String>().Utf8Value();
    Map* map = load_map_from_file(filename.c_str());
    
    if (map == nullptr) {
        Napi::Error::New(env, "Failed to load map from file").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    Napi::Object result = map_to_object(env, map);
    delete map;
    
    return result;
}

Napi::Value SaveMapToFile(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected 2 arguments").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Map* map = object_to_map(info[0].As<Napi::Object>());
    std::string filename = info[1].As<Napi::String>().Utf8Value();
    
    bool success = save_map_to_file(map, filename.c_str());
    delete map;
    
    return Napi::Boolean::New(env, success);
}

// ============================================================================
// Module Initialization
// ============================================================================

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
    ParticleSlamWrapper::Init(env, exports);
    
    exports.Set("load_map_from_file", Napi::Function::New(env, LoadMapFromFile));
    exports.Set("save_map_to_file", Napi::Function::New(env, SaveMapToFile));
    
    return exports;
}

NODE_API_MODULE(pswarm, InitAll)
