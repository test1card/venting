# Venting v10.0.0 — 0D/Network model of depressurization (rigid volumes + compressible discharge)

Этот репозиторий моделирует стравливание (depressurization / venting) системы жёстких объёмов газа, соединённых отверстиями/каналами, когда внешнее давление падает по заданному профилю `P_ext(t)` (`linear`, `step`, `barometric`, `table` из CSV).

Главный инженерный вопрос: какой максимальный перепад давления `|ΔP|` возникает на каждом «узком месте» (выход наружу, межузловые интерфейсы), и в каком режиме течёт газ (choked/subsonic).

**Политика версий:** в runtime активна только **v10.0.0**. Старый монолитный скрипт хранится только в `archive/` как архивный референс.

---

## 0) Интуиция для человека вне сферы

Есть несколько полостей с воздухом, соединённых отверстиями. Снаружи давление падает. Если наружное давление падает быстрее, чем газ успевает выйти, внутри какое-то время остаётся более высокое давление — появляется перепад давления на стенках/перегородках.

Типовая топология в проекте:

```text
N_parallel chains (aggregated):

cell10 -> cell9 -> ... -> cell2 -> cell1
   \                               |
    \______________________________|  (по N_parallel идентичных ветвей)
                                    v
                               [ vestibule ] --(exit orifice/slot)--> [ external P_ext(t) ]
```

Практически это нужно, чтобы оценить:
- где и когда пик `|ΔP|`,
- какие отверстия/`C_d` ограничивают продувку,
- насколько тепловой режим влияет на пик.

---

## 1) Что такое 0D/сетевая модель и чем она **не** является

### Что модель делает
- Каждый объём = один узел (`node`) с однородными `m(t), T(t), P(t)`.
- Связи между узлами = рёбра (`edges`) с моделью расхода.
- Модель собирается «по рёбрам»: вклад каждого ребра суммируется в балансы узлов.

### Что модель **не** делает
- Это не CFD: нет пространственных полей скорости/температуры.
- Нет акустики/ударных волн/3D-деталей струи.
- `C_d` задаётся как параметр (главный источник неопределённости).

---

## 2) Переменные и уравнение состояния

Для узла `i`:
- объём `V_i` (жёсткий),
- масса `m_i(t)`,
- температура `T_i(t)` (в режиме `intermediate`),
- давление из идеального газа:

$$
P_i V_i = m_i R T_i
$$

В v10.0.0 интегрируются `m` (и `T` в thermal-варианте), а давление вычисляется через EOS.

---

## 3) Массовый баланс

Для каждого узла:

$$
\frac{dm_i}{dt}=\sum \dot m_{in}-\sum \dot m_{out}
$$

Направление потока на ребре определяется по текущему давлению upstream/downstream.

---

## 4) Энергия и температура (почему blowdown охлаждает газ)

Для жёсткого контрольного объёма:

$$
\frac{d}{dt}(m c_v T)=\sum \dot m_{in} c_p T_{in}-\sum \dot m_{out} c_p T + \dot Q_{wall}
$$

Эквивалентная форма, используемая в коде:

$$
m c_v \frac{dT}{dt} = \sum \dot m_{in}(c_p T_{in}-c_vT) - \sum \dot m_{out}(RT) + hA_{wall}(T_{wall}-T)
$$

Режимы:
- `isothermal`: `T = const`, решается только масса.
- `intermediate`: решаются `m(t), T(t)`; при `h→0` поведение стремится к адиабатическому blowdown, при `h→∞` — к изотерме.

> Важно: в формулах расхода и энергетики используется **upstream temperature** из состояния узла.

---

## 5) Расход через отверстия, choking и отношение давлений

Для ребра:

$$
r = \frac{P_{down}}{P_{up}}, \quad r_* = \left(\frac{2}{\gamma+1}\right)^{\gamma/(\gamma-1)}
$$

- если `r <= r*` → **choked**,
- иначе → **subsonic**.

Это критично для пиков `|ΔP|`: choked-участок часто становится «бутылочным горлом» по массовому расходу.

---

## 6) Архитектура кода (активная v10.0.0)

Пакет: `src/venting/`

- `constants.py` — термоконстанты, safety-пороги, критическое отношение давлений.
- `geometry.py` — конвертация единиц, геометрические helper'ы.
- `profiles.py` — профили `P_ext(t)` (`linear/step/barometric/table`) и события profile-breakpoints.
- `graph.py` — узлы/рёбра/BC, построение сети `build_branching_network`.
- `flow.py` — расход через orifice и slot.
- `solver.py` — ODE RHS, `solve_ivp(method="Radau")`, события остановки, sparsity.
- `diagnostics.py` — `ΔP`, пики, режимы, `tau_exit`, meta.
- `gates.py` — встроенные gate checks (single-node / two-node).
- `io.py` — единообразная запись `run.json`, `summary.csv`, meta.
- `plotting.py` — опциональные графики.
- `cli.py` / `__main__.py` — CLI и точка входа `python -m venting`.

---

## 7) Численный решатель

- используется `scipy.integrate.solve_ivp`, метод **Radau** (жёсткие ODE),
- учитывается sparsity-структура,
- используются события ранней остановки (например, близость к внешнему давлению/низкое давление),
- **state clipping для P/T не используется**; есть только safety-защита знаменателей (`T_SAFE`, малые массы в делении).

---

## 8) Validation / Gate tests

В проекте есть детерминированные тесты `tests/test_gates.py`:

1. `test_single_node_analytic_match_adiabatic()`
   - один объём, выход в вакуум, `h=0`,
   - сравнение с аналитикой adiabatic blowdown,
   - маска `P > 0.01*P0`, точность < 0.5%.

2. `test_single_node_analytic_match_isothermal_limit()`
   - `h=1e6`, сравнение с изотермической экспонентой,
   - точность < 0.5%.

3. `test_mass_conservation_single_node()`
   - интеграл расхода согласован с `m0 - mf` (<0.1%).

4. `test_two_node_mass_conservation()`
   - масса по сети (<0.1%),
   - energy check для `h=0` (<1%).

5. `test_monotonic_pressure_when_vacuum()`
   - при `P_ext=0` давление не должно расти (кроме крошечного численного шума).

Запуск:

```bash
pytest -q
# или
python -m venting gate
python -m venting gate --single
python -m venting gate --two
```

---

## 9) Установка и запуск CLI

### Установка (рекомендуемо)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Альтернатива без dev-инструментов:

```bash
pip install -e .
```

### Основные команды

```bash
python -m venting --help
python -m venting gate
python -m venting sweep --profile linear --d-int 2 --d-exit 2 --cd-int 0.62 --cd-exit 0.62
python -m venting thermal --profile linear --d 2 --h-list 0,1,5,15
python -m venting sweep2d --profile linear --d-int-list 1.5,2.0 --d-exit-list 1.5,2.0
python -m venting sweep --profile linear --d-int 2 --d-exit 4 --int-model short_tube --L-int-mm 1 --exit-model short_tube --L-exit-mm 1 --K-in-int 0.5 --K-out-int 1.0 --eps-int-um 0 --K-in-exit 0.5 --K-out-exit 1.0 --eps-exit-um 0 --cd-int 0.62 --cd-exit 0.62 --thermo variable --wall-model lumped --external-model dynamic_pump --pump-speed-m3s 0.01 --V-ext 0.2 --P-ult-Pa 10
```

Профиль `table` ожидает CSV с колонками `t_s,P_Pa` через `--profile-file`.

---

## 10) How to interpret outputs

Каждый запуск пишет в:

```text
results/<timestamp>_<case>/
```

Типовые артефакты:
- `run.json` — воспроизводимость (timestamp, commit hash, python/platform, параметры, solver settings).
- `summary.csv` — ключевые метрики по рёбрам (`max_abs_dP_Pa`, `t_peak_s`, `r_peak`, `regime`, `peak_type`, `tau_exit_s`).
- `*.npz` — временные ряды (`t, m, T, P, P_ext, tau_exit`).
- `*_meta.json` — доп. метаданные.

Интерпретация ключевых полей:
- `r_peak = P_down/P_up` в момент пика,
- `regime`: `CHOKED` или `subsonic` (для slot — `viscous_slot`),
- `peak_type`:
  - `boundary(...)` — пик привязан к событию профиля `P_ext(t)`,
  - `internal` — пик рождается внутренней динамикой сети.
- `t_peak/tau_exit` (в meta/diagnostics) удобно для сравнения разных кейсов.

---

## 11) Limitations & red flags

Дополнительно для short-tube: это **lossy-nozzle** через эффективный `Cd_eff`; Fanno/friction-choking в v10.0.0 не реализован.



- Это **не** CFD и не замена стендовым испытаниям.
- `C_d` — главный источник неопределённости → обязательно делать sweep по `C_d`/геометрии.
- Изотермический случай не «всегда консервативен» для любой метрики и любого момента времени.
- Если видите странные осцилляции/нефизичные тренды — сначала проверяйте gate tests.

---

## Дополнительные документы

- `docs/model_v85.md` — краткая модель.
- `docs/verification_v85.md` — верификация и критерии.
- `docs/assumptions_v9.md` — допущения и ограничения.
- `CONTRIBUTING.md` — вклад в проект.
- `CHANGELOG.md` — история изменений.

## License

MIT (см. `LICENSE`).


## GUI

V10 adds an optional desktop GUI (thin client over the same core physics modules).

Install GUI extras:

```bash
pip install -e ".[gui]"
```

Run GUI:

```bash
python -m venting gui
```

What GUI supports:
- orifice vs short_tube for internal/exit edges (L, eps, K_in/K_out),
- network and geometry controls (`N_chain`, `N_par`, volumes, wall areas, diameters/counts, Cd),
- external model (`profile` / `dynamic_pump`),
- thermo (`isothermal` / `intermediate` / `variable`) and wall model (`fixed` / `lumped`),
- background solve with live plot updates, stop button, validity table,
- case JSON save/load (explicit units),
- artifact export compatible with CLI (`npz`, `meta.json`, `*_validity.json`, `summary.csv`).
- GUI progress uses chunked integration when streaming is enabled; tiny differences vs batch solve can appear within integrator tolerances.

Packaging executable for a target OS is left as TODO until `{TARGET_OS}` and `{PACKAGING_TOOL}` are specified.

GUI streaming uses chunked integration for progress updates; results can differ slightly from a single batch solve within integrator tolerances.
