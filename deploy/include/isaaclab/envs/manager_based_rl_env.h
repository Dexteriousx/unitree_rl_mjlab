// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>
#include <cstdio>
#include <cmath>
#include "isaaclab/utils/utils.h"

namespace isaaclab
{

// DEBUG INSTRUMENTATION (temporary, for B_DadDance NaN investigation):
// set to N by a caller (e.g. State_Mimic::enter) to dump full obs/action
// vectors for the next N calls to ManagerBasedRLEnv::step().
inline int g_debug_log_remaining = 0;
inline int g_debug_log_call_no = 0;

class ObservationManager;
class ActionManager;

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = cfg["step_dt"].as<float>();
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        robot->update();

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        robot->update();
        action_manager->reset();
        observation_manager->reset();
    }

    void step()
    {
        episode_length += 1;
        robot->update();
        auto obs = observation_manager->compute();
        auto action = alg->act(obs);

        if (g_debug_log_remaining > 0)
        {
            g_debug_log_call_no++;
            bool obs_bad = false; int obs_bad_idx = -1;
            int flat_idx = 0;
            for (auto & kv : obs) {
                for (float v : kv.second) {
                    if (std::isnan(v) || std::isinf(v)) { obs_bad = true; if (obs_bad_idx < 0) obs_bad_idx = flat_idx; }
                    flat_idx++;
                }
            }
            bool act_bad = false; int act_bad_idx = -1;
            for (size_t i = 0; i < action.size(); ++i) {
                if (std::isnan(action[i]) || std::isinf(action[i])) { act_bad = true; if (act_bad_idx < 0) act_bad_idx = (int)i; }
            }
            fprintf(stderr, "[MIMIC_DBG] === call #%d === obs_bad=%d(first_idx=%d) act_bad=%d(first_idx=%d) episode_length=%ld\n",
                    g_debug_log_call_no, (int)obs_bad, obs_bad_idx, (int)act_bad, act_bad_idx, episode_length);
            for (auto & kv : obs) {
                fprintf(stderr, "[MIMIC_DBG] OBS[%s] dim=%zu:", kv.first.c_str(), kv.second.size());
                for (float v : kv.second) fprintf(stderr, " %.6f", v);
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "[MIMIC_DBG] ACT dim=%zu:", action.size());
            for (float v : action) fprintf(stderr, " %.6f", v);
            fprintf(stderr, "\n");
            g_debug_log_remaining--;
        }

        action_manager->process_action(action);
    }

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    long episode_length = 0;
    float global_phase = 0.0f;
};

};