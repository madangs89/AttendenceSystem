import express, { type Request, type Response } from "express";
import dotenv from "dotenv";
import cors from "cors";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import joinImages from "join-images";
import fs from "fs";
dotenv.config();
const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const PY_URL = process.env.PY_URL ?? "http://127.0.0.1:8001";
const EMBED_SECRET = process.env.EMBED_SECRET ?? "secret";

app.use(
  cors({
    origin: "*",
  }),
);

const DB_FILE = "students.json";

type Student = {
  class: string;
  name: string;
  rollNo: string;
  embeddings: Partial<Record<faceAngles, number[]>>;
};

const loadDb = (): Student[] =>
  fs.existsSync(DB_FILE) ? JSON.parse(fs.readFileSync(DB_FILE, "utf8")) : [];
const saveDb = (db: Student[]) => fs.writeFileSync(DB_FILE, JSON.stringify(db));

const loadAttendance = (classId: string): string[] => {
  const fileName = "attendance.json";
  if (!fs.existsSync(fileName)) {
    return [];
  }
  const data = JSON.parse(fs.readFileSync(fileName, "utf8"));
  // normalize to strings so 1 and "1" are treated the same
  return (data[classId] || []).map(String);
};

const saveAttendance = (classId: string, attendance: string[]) => {
  const fileName = "attendance.json";

  const data: Record<string, string[]> = fs.existsSync(fileName)
    ? JSON.parse(fs.readFileSync(fileName, "utf8"))
    : {};

  data[classId] = [
    ...new Set([
      ...(data[classId] || []).map(String),
      ...attendance.map(String),
    ]),
  ];

  fs.writeFileSync(fileName, JSON.stringify(data, null, 2));
};

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 15 * 1024 * 1024,
  }, // 15 MB file size limit
});

type faceDetectionResult = {
  bbox: number[];
  det_score: number;
  embedding: number[];
};

// a detected face + which uploaded photo it came from
type TaggedFace = faceDetectionResult & { photoIdx: number };

const detectFace = async (
  file: Express.Multer.File,
): Promise<faceDetectionResult[]> => {
  const form = new FormData();
  form.append("file", file.buffer, {
    filename: file.originalname,
    contentType: file.mimetype,
  });
  const { data } = await axios.post(`${PY_URL}/embed`, form, {
    headers: { ...form.getHeaders(), "x-secret": EMBED_SECRET },
    maxBodyLength: Infinity,
    timeout: 60_000,
  });
  return data.faces;
};

type faceAngles = "front_face" | "left_face" | "right_face";

const faceUploadFields = upload.fields([
  { name: "front_face", maxCount: 1 },
  { name: "left_face", maxCount: 1 },
  { name: "right_face", maxCount: 1 },
]);

const dot = (a: number[], b: number[]) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
};

/**
 * Does all the matching work in one place:
 *  - matches faces to enrolled students (present / review)
 *  - counts UNIQUE faces in the photo(s), enrolled or not
 *  - builds every list the API returns
 */
function matchFaces(
  faces: TaggedFace[],
  gallery: Student[],
  markedAttendance: string[],
  threshold = 0.4,
  margin = 0.05,
  sameThreshold = 0.4, // face-vs-face threshold for "same unknown person"
) {
  const names = new Map(gallery.map((s) => [s.rollNo, s.name]));
  const everyOne = [...names.keys()];

  const candidates: { faceIdx: number; studentId: string; score: number }[] =
    [];
  const review: { studentId: string; score: number; reason: string }[] = [];

  // per-face info, used later for counting unique faces
  const faceInfo: {
    face: TaggedFace;
    status: "matched" | "review" | "none";
    studentId?: string;
  }[] = [];

  faces.forEach((face, faceIdx) => {
    // best score per student (max over that student's photos / angles)
    const best = new Map<string, number>();
    for (const g of gallery) {
      for (const [angle, embedding] of Object.entries(g.embeddings)) {
        if (!embedding) continue;
        const s = dot(face.embedding, embedding);
        if (s > (best.get(g.rollNo) ?? -1)) best.set(g.rollNo, s);
      }
    }
    const [top, second] = [...best.entries()].sort((a, b) => b[1] - a[1]);
    if (!top) {
      faceInfo.push({ face, status: "none" });
      return;
    }

    if (top[1] < threshold - 0.1) {
      // clearly not anyone enrolled
      faceInfo.push({ face, status: "none" });
      return;
    }

    if (top[1] < threshold) {
      review.push({ studentId: top[0], score: top[1], reason: "low score" });
      faceInfo.push({ face, status: "review", studentId: top[0] });
      return;
    }
    if (second && top[1] - second[1] < margin) {
      review.push({ studentId: top[0], score: top[1], reason: "ambiguous" });
      faceInfo.push({ face, status: "review", studentId: top[0] });
      return;
    }
    candidates.push({ faceIdx, studentId: top[0], score: top[1] });
    faceInfo.push({ face, status: "matched", studentId: top[0] });
  });

  // one face -> one student, highest scores win
  candidates.sort((a, b) => b.score - a.score);
  const usedFaces = new Set<number>();
  const present = new Map<string, number>();
  for (const c of candidates) {
    if (usedFaces.has(c.faceIdx) || present.has(c.studentId)) continue;
    usedFaces.add(c.faceIdx);
    present.set(c.studentId, c.score);
  }

  // ---------- UNIQUE FACE COUNT (enrolled + unknown) ----------
  // Step 1: faces confidently matched to a student are identified by the
  // student, not by the embedding. This handles the same person showing a
  // different angle in a different photo (their embeddings differ, but the
  // gallery has several angles, so both still match the same student).
  // Same student in the same photo twice = two different people, so for each
  // student the count is the max number of faces matched in ONE photo.
  const studentPhotoCount = new Map<string, Map<number, number>>();
  for (const info of faceInfo) {
    if (info.status !== "matched") continue;
    const perPhoto = studentPhotoCount.get(info.studentId!) ?? new Map();
    perPhoto.set(
      info.face.photoIdx,
      (perPhoto.get(info.face.photoIdx) ?? 0) + 1,
    );
    studentPhotoCount.set(info.studentId!, perPhoto);
  }
  let uniqueFacesCount = 0;
  for (const perPhoto of studentPhotoCount.values()) {
    uniqueFacesCount += Math.max(...perPhoto.values());
  }

  // Step 2: weak (review) faces that point to a student already identified in
  // another photo are the same person at a harder angle -> don't count again.
  const unknownFaces: TaggedFace[] = [];
  for (const info of faceInfo) {
    if (info.status === "matched") continue;
    if (info.status === "review") {
      const perPhoto = studentPhotoCount.get(info.studentId!);
      if (perPhoto && !perPhoto.has(info.face.photoIdx)) {
        perPhoto.set(info.face.photoIdx, 1); // claim it so it merges only once
        continue;
      }
    }
    unknownFaces.push(info.face);
  }

  // Step 3: cluster the remaining unknown faces. A face joins a cluster if it
  // is similar to ANY member (so a chain front -> half-left -> left still
  // links up), but never to a cluster already containing a face from its own
  // photo (two faces in one photo are always different people).
  const clusters: { members: TaggedFace[]; photos: Set<number> }[] = [];
  for (const face of unknownFaces) {
    let bestCluster: (typeof clusters)[number] | null = null;
    let bestScore = sameThreshold;
    for (const c of clusters) {
      if (c.photos.has(face.photoIdx)) continue;
      for (const m of c.members) {
        const s = dot(face.embedding, m.embedding);
        if (s >= bestScore) {
          bestScore = s;
          bestCluster = c;
        }
      }
    }
    if (bestCluster) {
      bestCluster.members.push(face);
      bestCluster.photos.add(face.photoIdx);
    } else {
      clusters.push({ members: [face], photos: new Set([face.photoIdx]) });
    }
  }
  uniqueFacesCount += clusters.length;

  const detectedFaces = new Set([
    ...present.keys(),
    ...review.map((r) => r.studentId),
  ]);

  // score + status for every detected student (confident or review)
  const detectionInfo = new Map<
    string,
    { score: number; status: "present" | "review"; reason?: string }
  >();
  for (const r of review) {
    const prev = detectionInfo.get(r.studentId);
    if (!prev || r.score > prev.score) {
      detectionInfo.set(r.studentId, {
        score: r.score,
        status: "review",
        reason: r.reason,
      });
    }
  }
  for (const [id, score] of present) {
    detectionInfo.set(id, { score, status: "present" }); // confident match wins
  }

  // marked but not detected => proxy
  const markedButNotDetected = markedAttendance
    .filter((id) => !detectedFaces.has(id))
    .map((id) => ({ rollNo: id, name: names.get(id) ?? "Unknown" }));

  // not marked but detected => inside class, teacher can call them to verify
  const notMarkedButDetected = [...detectedFaces]
    .filter((id) => !markedAttendance.includes(id))
    .map((id) => {
      const info = detectionInfo.get(id);
      return {
        rollNo: id,
        name: names.get(id) ?? "Unknown",
        score: info ? Number(info.score.toFixed(3)) : null,
        status: info?.status ?? null, // "present" = confident, "review" = weak match
        reason: info?.reason ?? null, // only for review: "low score" / "ambiguous"
      };
    });

  // marked and detected
  const presentList = [...present.entries()]
    .filter(([id]) => markedAttendance.includes(id))
    .map(([id, score]) => ({
      studentId: id,
      name: names.get(id),
      score: Number(score.toFixed(3)),
    }));

  // students needing teacher review (and not already confirmed present)
  const reviewList = review
    .filter(
      (r) =>
        !present.has(r.studentId) && markedAttendance.includes(r.studentId),
    )
    .map((r) => ({
      studentId: r.studentId,
      name: names.get(r.studentId),
      score: Number(r.score.toFixed(3)),
      reason: r.reason,
    }));

  // absent: not detected in the photo AND not marked present
  const unconfirmed = everyOne
    .filter((id) => !detectedFaces.has(id) && !markedAttendance.includes(id))
    .map((id) => ({ studentId: id, name: names.get(id) }));

  return {
    uniqueFacesCount, // every distinct person in the photo(s)
    enrolledDetectedCount: detectedFaces.size, // only enrolled students recognised
    unknownFacesCount: Math.max(0, uniqueFacesCount - detectedFaces.size),
    presentList,
    reviewList,
    notMarkedButDetected,
    markedButNotDetected,
    unconfirmed,
  };
}

app.get("/", (req: Request, res: Response) => {
  res.send("<h1>Hello, World!</h1>");
});

app.post(
  "/mark-present",
  async (
    req: Request<{}, {}, { classId: string; studentId: string | number }>,
    res,
  ) => {
    try {
      const { classId } = req.body;
      const studentId =
        req.body.studentId !== undefined && req.body.studentId !== null
          ? String(req.body.studentId)
          : "";

      if (!classId || !studentId) {
        return res.status(400).json({
          success: false,
          message: "Missing fields (classId , studentId)",
        });
      }

      const markedAttendance = loadAttendance(classId);

      if (!markedAttendance.includes(studentId)) {
        markedAttendance.push(studentId);
        saveAttendance(classId, markedAttendance);
        return res
          .status(200)
          .json({ message: "Attendance marked successfully", success: true });
      } else {
        return res
          .status(200)
          .json({ message: "Attendance already marked", success: true });
      }
    } catch (error) {
      console.log("error while marking attendance", error);

      return res
        .status(500)
        .json({ message: "Error marking attendance", success: false });
    }
  },
);

app.post(
  "/enroll",
  faceUploadFields,
  async (req: Request<{}, {}, Student>, res: Response) => {
    try {
      const { class: studentClass, name, rollNo } = req.body;
      if (!studentClass || !name || !rollNo) {
        return res
          .status(400)
          .json({ message: "Missing required fields", success: false });
      }
      const file = req.files as
        Partial<Record<faceAngles, Express.Multer.File[]>> | undefined;

      if (!file || Object.keys(file).length === 0) {
        return res
          .status(400)
          .json({ message: "No file uploaded", success: false });
      }

      const getFile = (
        fieldName: faceAngles,
      ): Express.Multer.File | undefined => {
        const files = file[fieldName] as Express.Multer.File[] | undefined;
        return files ? files[0] : undefined;
      };

      // keys now match faceAngles exactly
      const allFaces: Record<faceAngles, Express.Multer.File | undefined> = {
        front_face: getFile("front_face"),
        left_face: getFile("left_face"),
        right_face: getFile("right_face"),
      };

      if (!allFaces.front_face || !allFaces.left_face || !allFaces.right_face) {
        return res.status(400).json({
          message: "Missing one or more face images (front, left, right)",
          success: false,
        });
      }

      const db: Student = {
        class: studentClass,
        name,
        rollNo,
        embeddings: {},
      };

      for (const [key, value] of Object.entries(allFaces) as [
        faceAngles,
        Express.Multer.File | undefined,
      ][]) {
        if (!value) {
          return res.status(400).json({
            message: `Missing ${key} face image`,
            success: false,
          });
        }
        const faces: faceDetectionResult[] = await detectFace(value);

        if (faces.length === 0) {
          return res.status(400).json({
            message: `No face detected in ${key} face image`,
            success: false,
          });
        }

        if (faces.length > 1) {
          return res.status(400).json({
            message: `Multiple faces detected in ${key} face image`,
            success: false,
          });
        }

        db.embeddings[key] = faces[0]!.embedding;
      }

      const storage = loadDb();
      storage.push(db);
      saveDb(storage);
      return res.status(200).json({
        message: "Student enrolled successfully",
        success: true,
        data: db,
      });
    } catch (error) {
      console.error("Error enrolling student:", error);
      res
        .status(500)
        .json({ message: "Internal Server Error", success: false });
    }
  },
);

app.post(
  "/mark-attendance",
  upload.array("class", 15),
  async (req: Request<{}, {}, { classId: string }>, res: Response) => {
    try {
      // accept body and file
      const { classId } = req.body;
      if (!classId) {
        return res
          .status(400)
          .json({ message: "Missing required fields", success: false });
      }

      // handle no file case
      const files = req.files as Express.Multer.File[];
      console.log(
        "files received:",
        files?.length,
        files?.map((f) => f.originalname),
      );

      if (!files || files.length === 0) {
        return res
          .status(400)
          .json({ message: "Class Photo is not Provided", success: false });
      }

      // get the face embeddings from every photo (tagged with their photo index)
      const allExtractedFaces: TaggedFace[] = [];
      const facesPerPhoto: { photo: string; facesFound: number }[] = [];

      for (const [photoIdx, file] of files.entries()) {
        const faces: faceDetectionResult[] = await detectFace(file);
        facesPerPhoto.push({
          photo: file.originalname,
          facesFound: faces.length,
        });

        // skip photos with no face instead of failing the whole request
        if (faces.length === 0) continue;

        allExtractedFaces.push(...faces.map((f) => ({ ...f, photoIdx })));
      }

      if (allExtractedFaces.length === 0) {
        return res.status(400).json({
          message: "No face detected in any of the images",
          success: false,
          photosProcessed: files.length,
          facesPerPhoto,
        });
      }

      // sort faces by detection score in descending order
      const faces = allExtractedFaces.sort((a, b) => b.det_score - a.det_score);

      // students of this class from the database
      const gallery = loadDb().filter((s) => s.class === classId);

      if (!gallery || gallery.length === 0) {
        return res.status(400).json({
          message: "No students enrolled in the class",
          success: false,
        });
      }

      const markedAttendance = loadAttendance(classId);

      const {
        uniqueFacesCount,
        enrolledDetectedCount,
        unknownFacesCount,
        presentList,
        reviewList,
        notMarkedButDetected,
        markedButNotDetected,
        unconfirmed,
      } = matchFaces(faces, gallery, markedAttendance);

      return res.status(200).json({
        message: "Attendance marked successfully",
        success: true,
        photosProcessed: files.length,
        facesPerPhoto,
        detectedFacesCount: uniqueFacesCount, // all unique faces in the photo(s)
        enrolledDetectedCount, // recognised enrolled students
        unknownFacesCount, // faces that aren't enrolled students
        markedAttendanceCount: markedAttendance.length,
        present: presentList,
        notMarkedButDetected,
        review: reviewList,
        unconfirmed,
        markedButNotDetected,
      });
    } catch (error) {
      console.error("Error marking attendance:", error);
      res
        .status(500)
        .json({ message: "Error marking attendance", success: false });
    }
  },
);

app.listen(3000, () => {
  console.log("Server is running on port 3000");
});
