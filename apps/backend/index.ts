import express, { type Request, type Response } from "express";
import dotenv from "dotenv";
import cors from "cors";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
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

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 15 * 1024 * 1024,
  }, // 15 MB file size limit
});

app.get("/", (req: Request, res: Response) => {
  res.send("<h1>Hello, World!</h1>");
});

type faceDetectionResult = {
  bbox: number[];
  det_score: number;
  embedding: number[];
};

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

      let allFaces = {
        front: getFile("front_face"),
        left: getFile("left_face"),
        right: getFile("right_face"),
      };

      if (!allFaces.front || !allFaces.left || !allFaces.right) {
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

      for (const [stringKey, value] of Object.entries(allFaces)) {
        const key = stringKey as faceAngles;
        if (!value) {
          continue;
        }
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
    // accept body and file
    const { classId } = req.body;

    if (!classId) {
      return res
        .status(400)
        .json({ message: "Missing required fields", success: false });
    }
    // handle no file case
    const files = req.files as Express.Multer.File[];

    if (!files || files.length == 0) {
      return res
        .status(400)
        .json({ message: "Class Photo is not Provided", success: false });
    }
    // get the face embeddings from the file
    // compare the embeddings with the stored embeddings in the database for a particular class
    // Need to check across 3 facial angles
    // Edge cases
    //      1.No face detected in the image
    //      2.
    // return the response with the attendance status
  },
);

app.listen(3000, () => {
  console.log("Server is running on port 3000");
});
