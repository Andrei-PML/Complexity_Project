import math
import random
import datetime
import matplotlib.pyplot as plt



class Student:

    ZPD_BELOW = 0.10    # challenge this far *below* current knowledge -> too easy, no growth
    ZPD_ABOVE = 0.25    # challenge this far *above* current knowledge -> too hard, no growth

    ATTENTION_DECAY       = 0.003   # passive attention decay per step
    ATTENTION_NOISE_SCALE = 0.012   # sd of random normal noise added each step
    DISTRACTOR_MAX_DIST   = 2.0     # max distance at which distractors have an effect
    ATTENTION_THRESHOLD   = 0.30    # attention below this -> student becomes a distractor

    KNOWLEDGE_RATE = 0.005  # base knowledge gain per step when attentive and in ZPD
    PEER_BONUS     = 0.002  # bonus per attentive neighbour within PEER_MAX_DIST seats
    PEER_MAX_DIST  = 1.5    # seat distance threshold for peer interaction

    def __init__(self, student_id, row, col, teacher_boost, distractor_effect, knowledge=random.uniform(0.0, 0.4), attention=random.uniform(0.4, 0.9)):
        self.student_id    = student_id
        self.row           = row
        self.col           = col
        self.knowledge     = knowledge
        self.attention     = attention
        self.is_distractor = self.attention < self.ATTENTION_THRESHOLD

        self.teacher_boost = teacher_boost
        self.distractor_effect = distractor_effect


    def update_attention(self, teacher_share, neighbours):
        """
        teacher_share: float - This student's fraction of total teacher attention.
        neighbours: list of (SnapStudent, float)
            (snapshot_proxy, seat_distance) pairs for every other student.
            Distractor flags are frozen to the start-of-step snapshot so all
            updates are synchronous.
        """
        delta = 0.0

        # Teacher attention provides a stabilising boost
        delta += teacher_share * self.teacher_boost

        # Passive attentional decay
        delta -= self.ATTENTION_DECAY

        # Nearby distractors reduce attention; effect drops off with distance and is capped at DISTRACTOR_MAX_DIST
        for other, dist in neighbours:
            if other.is_distractor and 0 < dist <= self.DISTRACTOR_MAX_DIST:
                delta -= self.distractor_effect / dist

        # Random variation. Comment to reduce stochasticity:
        delta += random.gauss(0.0, self.ATTENTION_NOISE_SCALE)

        self.attention     = max(0.0, min(1.0, self.attention + delta))
        self.is_distractor = self.attention < self.ATTENTION_THRESHOLD


    def update_knowledge(self, challenge_level, neighbours):
        """
        Advance knowledge if the student is attentive and challenge is in ZPD.

        challenge_level: float - current environment challenge level in [0, 1]
        neighbours: list of (SnapStudent, float)
        """
        if self.is_distractor:
            return  # distracted students do not learn

        diff = challenge_level - self.knowledge
        in_zpd = (-self.ZPD_BELOW <= diff <= self.ZPD_ABOVE)
        if not in_zpd:
            return

        gain = self.KNOWLEDGE_RATE

        # Peer interaction bonus from attentive nearby classmates
        for other, dist in neighbours:
            if not other.is_distractor and dist <= self.PEER_MAX_DIST:
                gain += self.PEER_BONUS

        self.knowledge = min(1.0, self.knowledge + gain)


    def seat_distance(self, other):

        return math.dist((self.row, self.col), (other.row, other.col))



class SnapStudent:
    """
    Frozen view of a student at the beginning of an iteration
    """
    __slots__ = ("is_distractor", "knowledge", "row", "col", "student_id")

    def __init__(self, student, snap_distractor):
        self.is_distractor = snap_distractor
        self.knowledge     = student.knowledge
        self.row           = student.row
        self.col           = student.col
        self.student_id    = student.student_id



class Classroom:

    CHALLENGE_GROWTH_RATE = 0.001   # increase in challenger per iteration
    CHALLENGE_NOISE_SCALE = 0.003   # sd of per-step challenge fluctuation

    def __init__(
        self,
        rows=4,
        cols=5,
        n_students=20,
        iterations=200,
        log_path="classroom_log.txt",
        teacher_boost=0.10,
        distractor_effect=0.6
    ):

        self.rows             = rows
        self.cols             = cols
        self.n_students       = n_students
        self.iterations       = iterations
        self.log_path         = log_path

        self.teacher_boost = teacher_boost
        self.distractor_effect = distractor_effect

        self.students         = []
        self.challenge_level  = 0.15
        self.iteration        = 0
        self._metrics_history = []
        self._log_lines       = []
        self._dist_cache      = {}   # keyed (student_id_a, student_id_b)

        self._place_students()
        self._precompute_distances()
        self._write_log_header()


    def _place_students(self):
        seats = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        random.shuffle(seats)
        for i in range(self.n_students):
            r, c = seats[i]
            self.students.append(Student(i, r, c, teacher_boost=self.teacher_boost, distractor_effect=self.distractor_effect))


    def _precompute_distances(self):
        """Seat distances are fixed - cache them once at init."""
        for s in self.students:
            for other in self.students:
                if s is not other:
                    self._dist_cache[(s.student_id, other.student_id)] = (s.seat_distance(other))


    def _distribute_teacher_attention(self):
        """Return {student_id: share} where all shares sum to 1.0."""
        n      = len(self.students)
        shares = {}

        weights = [max(0.0, 1.0 - s.knowledge) for s in self.students]
        total   = sum(weights) or 1.0
        for s, w in zip(self.students, weights):
            shares[s.student_id] = w / total

        return shares


    def step(self):
        """Advance the simulation by one iteration."""
        self.iteration += 1
        teacher_shares = self._distribute_teacher_attention()

        # Freeze distractor flags so all agents update synchronously
        distractor_snap = {s.student_id: s.is_distractor for s in self.students}

        # Build neighbour lists with snapshot proxies (distances are fixed)
        nbr_cache = {}
        for s in self.students:
            nbr_cache[s.student_id] = [
                (SnapStudent(other, distractor_snap[other.student_id]),
                 self._dist_cache[(s.student_id, other.student_id)])
                for other in self.students if other is not s]

        # Apply updates
        for s in self.students:
            s.update_attention(teacher_shares[s.student_id], nbr_cache[s.student_id])

        for s in self.students:
            s.update_knowledge(self.challenge_level, nbr_cache[s.student_id])

        # Update environment's challenge level
        self.challenge_level =  min(1.0,
                self.challenge_level
                + self.CHALLENGE_GROWTH_RATE
                + random.gauss(0.0, self.CHALLENGE_NOISE_SCALE))

        # Record mean knowledge
        self._record_metrics(sum(s.knowledge for s in self.students) / len(self.students))


    def _record_metrics(self, mean_knowledge):

        n_distractors  = sum(s.is_distractor for s in self.students)
        mean_attention = sum(s.attention for s in self.students) / self.n_students

        n_in_ZPD = 0
        for s in self.students:
            if -s.ZPD_BELOW <= self.challenge_level - s.knowledge <= s.ZPD_ABOVE:
                n_in_ZPD += 1
        percentage_in_ZPD = 100 * n_in_ZPD / self.n_students

        metrics = {
            "iteration"        : self.iteration,
            "mean_knowledge"   : round(mean_knowledge, 4),
            "mean_attention"   : round(mean_attention, 4),
            "n_distractors"    : n_distractors,
            "challenge_level"  : round(self.challenge_level, 4),
            "percentage_in_ZPD": round(percentage_in_ZPD, 2)
        }
        self._metrics_history.append(metrics)

        self._log(
            f"[{self.iteration:03d}]  "
            f"knowledge={metrics['mean_knowledge']:.4f}  "
            f"attention={metrics['mean_attention']:.4f}  "
            f"distractors={n_distractors:02d}/{self.n_students}  "
            f"challenge={metrics['challenge_level']:.4f}"
        )

    @property
    def metrics_history(self):
        return self._metrics_history


    def _write_log_header(self):

        self._log("=" * 80)
        self._log(f"Started       : {datetime.datetime.now().isoformat(timespec='seconds')}")
        self._log(f"Grid          : {self.rows} rows x {self.cols} cols")
        self._log(f"Students      : {self.n_students}")
        self._log(f"Iterations    : {self.iterations}")
        self._log("-" * 80)

    def _log(self, line):
        self._log_lines.append(line)

    def _flush_log(self):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(self._log_lines))
            f.write("\n")


    def run(self):

        for _ in range(self.iterations):
            self.step()

        final = self._metrics_history[-1]
        self._log("=" * 80)
        self._log("\n\n\n")
        self._flush_log()
        print(f"Simulation finished ({self.iteration} steps). Log: {self.log_path}")


if __name__ == "__main__":

    teacher_boos_values =      [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    distractor_effect_values = [0.0, 0.30, 0.60, 0.90, 1.20, 1.50]

    fig, axes = plt.subplots(6, 6, figsize=(18, 18), sharex=True, sharey=True)

    for i, tb in enumerate(teacher_boos_values):
        for j, de in enumerate(distractor_effect_values):

            sim = Classroom(
                rows=4,
                cols=5,
                n_students=20,
                iterations=200,
                log_path="classroom_log.txt",
                teacher_boost=tb,
                distractor_effect=de
            )
            sim.run()

            metrics = sim.metrics_history
            iteration = [m["iteration"] for m in metrics]
            percentage_in_ZPD = [m["percentage_in_ZPD"] for m in metrics]

            ax = axes[i, j]
            ax.plot(iteration, percentage_in_ZPD, color="black")
            ax.set_title(f"tb={tb:.2f}, de={de:.2f}", fontsize=8)

    fig.supxlabel("Time")
    fig.supylabel("Percentage of students in ZPD")
    fig.suptitle("Percentage of students in ZPD over time", fontsize=17)
    fig.tight_layout(pad=3)
    plt.show()
