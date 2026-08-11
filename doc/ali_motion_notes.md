# ali_motion (GMR → mjlab) — известные проблемы и команды

Заметки по прогону кастомного мотива (`ali_motion`, GMR retarget → G1) через
`unitree_rl_mjlab`. Цель — не наступать второй раз на одни и те же грабли.

## Пайплайн данных

```
GMR retarget (30 fps)                convert_gmr_to_mjlab_fk.py           mjlab MotionCommand
<path-to-GMR>/outputs/          -->  src/assets/motions/g1/          -->  тренировка / play.py
ali_motion_unitree_g1_ref.npz        ali_motion_unitree_g1_fk.npz
(root_pos, root_rot xyzw, dof_pos)   (joint_pos/vel, body_pos/quat/
                                       lin_vel/ang_vel_w @ 50fps)
```

Опубликованная версия (после ресемплинга/фиксов ниже) лежит в
`motions/ali_dance/motion.npz` — см. `motions/ali_dance/README.md`.

## Баги, которые нашли и исправили в `convert_gmr_to_mjlab_fk.py` (2026-08-07)

Все три бага объясняют, почему прошлые прогоны (`diag_fixed_lr3e4`,
`diag_adaptive_baseline`, `diag_entropy2x`, `diag_fixed_lr3e4_motionfix*`,
июль 2026) не заводились — проблема была в данных, не в гиперпараметрах.

1. **Лишнее тело `world` в body-массивах.** `body_pos_w/body_quat_w/
   body_lin_vel_w/body_ang_vel_w` сохранялись как `data.xpos`/`data.xquat`
   "как есть" — 31 тело (mujoco body 0 = world). `MotionLoader` в
   `mjlab`/`src/tasks/tracking/mdp/commands.py` индексирует эти массивы
   entity-относительными индексами (`Entity.body_names` = `spec.bodies[1:]`,
   world уже отброшен) — как в эталонном `dance1_subject2.npz` (30 тел).
   Результат: сдвиг на одно звено у ВСЕХ 14 отслеживаемых тел. root(`pelvis`)
   читался как `world` (constant identity quat `[1,0,0,0]`, нулевая
   скорость), anchor (`torso_link`) — как `waist_roll_link`.
   **Симптом-маркер**: root quat == `[1,0,0,0]` на всех кадрах.
   **Фикс**: сохранять `body_pos_w_full[:, 1:]` и т.д. (дропать world).

2. **Не было ресемплинга 30→50 fps.** Источник GMR — 30 fps, `MotionLoader`
   продвигает кадр на +1 каждый control-step (50 Hz = 1/(timestep=0.005 ×
   decimation=4)). Без ресемплинга мотив проигрывался в 1.67x быстрее
   задуманного (635 кадров за 12.7с вместо 21.2с).
   **Фикс**: линейная интерполяция root_pos/joint_pos + slerp для root_rot,
   30→50 fps (как в каноническом `scripts/csv_to_npz.py`, `input_fps=30,
   output_fps=50`). Ключ `fps` теперь тоже пишется в npz (для консистентности
   с `dance1_subject2.npz`, хотя `MotionLoader` его не читает).

3. **Немотивированный `root_pos[:, 2] += 0.2`.** Поднимал всё тело на 20 см,
   стопы никогда не касались пола (min z ≈ 0.24 м весь клип, у baseline
   dance1_subject2 — min z ≈ 0.02 м). **Убрали offset полностью** (по
   решению от 2026-08-07) — без него min foot z ≈ 0.04 м, что уже близко к
   baseline.

## Источник данных ali_motion

Полная цепочка получения `ali_motion_unitree_g1_ref.npz` (до конвертации
скриптом выше):

```
видео (21.2 с)  -->  GVHMR (SMPL/SMPLX pose)  -->  GMR gvhmr_to_robot.py  -->  ali_motion_unitree_g1_ref.npz
                      hmr4d_results.pt              --robot unitree_g1
```

- **GVHMR** (zju3dv/GVHMR, чекпоинт `gvhmr_siga24_release.ckpt`) извлекает
  позу человека (SMPL/SMPLX) из видео. Лицензия — **кастомная non-commercial
  research-only** (использование только в образовательных/исследовательских/
  некоммерческих целях; для коммерческого применения нужно отдельное
  разрешение правообладателя, ZJU3DV).
- **GMR** (`YanjieZe/GMR`, MIT) ретаргетит SMPL/SMPLX-позу на скелет G1
  (`scripts/gvhmr_to_robot.py --gvhmr_pred_file ... --robot unitree_g1`).
  MIT покрывает только код ретаргетера, не отменяет ограничение GVHMR на
  сами данные позы.
- **mjlab** — дальше motion воспроизводится/тренируется как обычный tracking
  motion (см. пайплайн выше).

**Следствие**: `ali_motion_unitree_g1_*.npz` и любая политика, обученная на
них (в т.ч. `motions/ali_dance/`), наследуют non-commercial research-only
ограничение GVHMR. Для коммерческого использования нужно либо получить
разрешение ZJU3DV, либо переретаргетить исходное видео через
non-restricted пайплайн (bvh/mocap без GVHMR).

## Чек-лист валидации нового motion-файла (до кинематического реплея/тренировки)

1. Форма массивов: `body_pos_w`/`body_quat_w`/... должны быть `(T, 30, ...)`
   — **без** world (сверять с `dance1_subject2.npz` как эталоном).
2. `fps` в файле (или фактическая частота кадров) == control rate env
   (`1 / (sim.timestep * decimation)`, сейчас 50 Hz для G1 tracking).
3. root quat на нескольких кадрах (0/25/50/100...) — не должен быть константой
   `[1,0,0,0]`.
4. `max abs(body_ang_vel_w)` — не 0, в пределах диапазона эталона
   (dance1_subject2: до ~37 рад/с).
5. `min` высоты стоп (`left/right_ankle_roll_link`, z) по всем кадрам — должна
   быть близка к 0 (эталон ~0.02–0.04 м), не на 10+ см выше.
6. `max abs(joint_vel)` — сверить с эталоном (dance1_subject2: до ~30 рад/с),
   резкие скачки на 1 кадр — повод посмотреть кадр глазами.
7. Кинематический реплей прямой установкой qpos (без физики) — визуально
   проверить позы, особенно вокруг подозрительных кадров из п.6.
8. **Дополнительно стоит симулировать реальную индексацию `MotionLoader`**
   (entity-relative индексы применяются к сырому массиву напрямую) — баг №1
   не находится просто чтением "сырых" данных по правильным индексам, нужно
   явно проверить, что читает сам loader.

Скрипт для генерации: `convert_gmr_to_mjlab_fk.py` (в корне репозитория;
`--src`/`--dst`/`--output-fps` — аргументы командной строки).

## Команды

### Активация окружения
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mjlab
cd <path-to-repo>/unitree_rl_mjlab
```

### Конвертация GMR → mjlab fk.npz
```bash
python convert_gmr_to_mjlab_fk.py \
  --src <path-to-GMR>/outputs/ali_motion_unitree_g1_ref.npz \
  --dst src/assets/motions/g1/ali_motion_unitree_g1_fk.npz
```

### Тренировка (эталонная конфигурация, как на dance1_subject2)
Task id: `Unitree-G1-Tracking` (src/tasks/tracking/config/g1/__init__.py).
wandb в оффлайне — иначе падает с `UsageError: No API key configured`.
```bash
WANDB_MODE=offline python scripts/train.py Unitree-G1-Tracking \
  --env.scene.num-envs 4096 \
  --motion-file src/assets/motions/g1/ali_motion_unitree_g1_fk.npz \
  --agent.max-iterations 2000 \
  --agent.run-name <name>
```
Дефолты из git (не переопределялись, совпадают с рабочим прогоном
`2026-08-07_15-01-41` на dance1_subject2): `learning_rate=1e-3`,
`schedule=adaptive`, `num_learning_epochs=5`, `num_mini_batches=4`,
`num_steps_per_env=24` → batch = num_envs × num_steps_per_env = 98304.

Финальный чекпоинт называется `model_<max_iterations - 1>.pt`
(0-индексация), например `model_1999.pt` для `--agent.max-iterations 2000` —
**не** `model_2000.pt`. Важно при мониторинге лога на "дошло до N".

### Просмотр обученной политики (живой вьювер)
GUI-сессия на `DISPLAY=:1` (не `:0` — проверять `who`/`loginctl`).
```bash
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
python scripts/play.py Unitree-G1-Tracking \
  --checkpoint-file logs/rsl_rl/g1_tracking/<run_dir>/model_<N>.pt \
  --motion-file src/assets/motions/g1/ali_motion_unitree_g1_fk.npz \
  --num-envs 1 \
  --viewer native
```
Процесс блокируется в render-loop до закрытия окна — это нормально, не баг.
Запускать через фоновый механизм инструментов (не `timeout ... &`, иначе
можно случайно убить окно таймаутом).

### Кинематический реплей (без физики, прямая установка qpos)
Скрипт-шаблон: строит qpos из `root_pos/root_rot(xyzw)/dof_pos`,
`mujoco.mj_forward`, рендер через `mujoco.Renderer` + `imageio`. Модель без
пола/света в сцене (используется чистый `g1.xml`) — для проверки контакта
стоп с полом полагаться на числа (min z), не на картинку.
