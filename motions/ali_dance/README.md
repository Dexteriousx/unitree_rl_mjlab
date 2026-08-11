# ali_dance

Кастомный tracking-мотив для Unitree G1, полученный из собственной видеозаписи
(не bvh/mocap): видео → [GVHMR](https://github.com/zju3dv/GVHMR) (SMPL/SMPLX
pose) → [GMR](https://github.com/YanjieZe/GMR) (`gvhmr_to_robot.py --robot
unitree_g1`) → `convert_gmr_to_mjlab_fk.py`. Полная цепочка и разбор багов,
найденных при конвертации, — в `doc/ali_motion_notes.md`.

- **Длительность**: 21.14 с (1057 кадров @ 50 fps)
- **Обучение**: task `Unitree-G1-Tracking`, 2000 итераций, `num_envs=4096`,
  запуск `ali_motion_fixed` (2026-08-07), дефолтный конфиг трекинга
  (`learning_rate=1e-3`, `schedule=adaptive`, `num_learning_epochs=5`,
  `num_mini_batches=4`, `num_steps_per_env=24`)

## Файлы

- `motion.npz` — retarget-мотив в формате mjlab (`joint_pos/vel`,
  `body_pos/quat/lin_vel/ang_vel_w`), готов для `--motion-file` в
  `scripts/train.py` / `scripts/play.py`.
- `policy.onnx` — финальный чекпоинт (`model_1999.pt`), экспортированный в
  ONNX для деплоя (см. `deploy/robots/g1/config/policy/mimic/`).
- `demo.mp4` — кинематический реплей мотива (прямая установка `qpos`, без
  физики/RL-политики) — визуальная проверка ретаргета, не прогон обученной
  политики в симуляторе.

## Лицензионное ограничение

Поза извлечена GVHMR, который распространяется по кастомной **non-commercial
research-only** лицензии (ZJU3DV). Это ограничение наследуют `motion.npz` и
`policy.onnx` в этой папке — использование только в образовательных/
исследовательских/некоммерческих целях. Для коммерческого применения нужно
отдельное разрешение правообладателя GVHMR. Подробности — в разделе
"Источник данных ali_motion" в `doc/ali_motion_notes.md`.
