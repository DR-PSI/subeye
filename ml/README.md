# SubEye · Daily ML บน GitHub Actions

เทรน LSTM ต่ออุปกรณ์ พยากรณ์ 24 ชั่วโมงข้างหน้า แล้วเขียนคะแนนกลับ Firestore ฟิลด์ `ml`
รันวันละครั้งบน runner ของ GitHub ไม่ต้องมีเซิร์ฟเวอร์เปิดค้าง

## ไฟล์

```
.github/workflows/subeye-daily-ml.yml   ตารางเวลา + ขั้นตอนรัน
ml/daily_ml.py                          สคริปต์หลัก
ml/requirements.txt                     dependencies
ml/artifacts/                           น้ำหนักโมเดล + รายงาน (สร้างอัตโนมัติ)
```

## ตั้งค่าครั้งเดียว

**1. Service account key**

Firebase Console → Project settings → Service accounts → Generate new private key
ได้ไฟล์ JSON มา แล้วไปที่ repo → Settings → Secrets and variables → Actions → New repository secret

| ชื่อ | ค่า |
|---|---|
| `FIREBASE_SERVICE_ACCOUNT` | วางเนื้อหาไฟล์ JSON ทั้งไฟล์ |
| `LINE_WEBHOOK_URL` | `https://n8n.jupetor-cmms.com/webhook/subeye-alarm` |

**2. ขอบเขตที่จะประมวลผล** (ไม่บังคับ)

แท็บ Variables → New repository variable ชื่อ `SUBEYE_SCOPES` ค่าเป็น JSON:

```json
[
  {"level":"substation","station":"คลองเตย","substation":"WTK"},
  {"level":"substation","station":"คลองเตย","substation":"BPT"},
  {"level":"district","station":"คลองเตย"}
]
```

ไม่ตั้งก็ได้ — สคริปต์มีค่า default 3 ขอบเขตนี้อยู่แล้ว

**3. ทดลองรัน**

แท็บ Actions → SubEye Daily ML → Run workflow → ติ๊ก **dry_run** → Run
โหมดนี้คำนวณให้ดูแต่ไม่เขียนอะไรกลับ Firestore ดู log ได้ว่าอุปกรณ์ไหนข้อมูลพอแล้ว

## โมเดลทำงานอย่างไร

สคริปต์ไม่ยัด LSTM ให้ทุกกรณี เลือกตามปริมาณข้อมูลที่มี:

| จุดใน `hourly` | ทำอะไร |
|---|---|
| < 48 (2 วัน) | `level: learning` ยังไม่ประเมิน |
| 48–335 | Holt-Winters additive เท่านั้น |
| ≥ 336 (14 วัน) | เทรน LSTM แล้ว**เทียบ MAE กับ Holt-Winters** ตัวไหนแม่นกว่าใช้ตัวนั้น |

ตัวที่ถูกใช้จริงบันทึกไว้ในฟิลด์ `ml.method` (`holt-winters` หรือ `lstm`) พร้อม `ml.mae` และ `ml.mae_baseline` ให้เทียบย้อนหลังได้ว่า LSTM คุ้มจริงไหม

โครงสร้าง LSTM: 2 ชั้น (32 → 16 units) + dropout 0.15, huber loss, early stopping, ดูย้อนหลัง 48 ชั่วโมงเพื่อทำนายชั่วโมงถัดไป แล้วพยากรณ์แบบ recursive 24 รอบ น้ำหนักที่เทรนแล้วถูก cache ไว้ รอบต่อไปเทรนต่อจากเดิม ไม่เริ่มใหม่

การจับความผิดปกติใช้ median + MAD ของ residual ไม่ใช่ mean + sd เพราะทนค่าโดดจากสัญญาณรบกวน RS485 ได้ดีกว่า

## คะแนน

เริ่มที่ 100 แล้วหัก:
- ความผิดปกติเทียบ baseline — หักได้ถึง 46
- ความเร่งเข้าหาเกณฑ์เตือน — หักได้ถึง 34
- ความคลาดเคลื่อนของโมเดลเอง — หักได้ถึง 10

`score ≥ 85` = ok · `70–84` = watch · `< 70` = risk

## ฟิลด์ที่เขียนกลับ

```
ml.score          คะแนน 0-100 (null ถ้ายังเรียนรู้)
ml.level          ok | watch | risk | learning
ml.method         holt-winters | lstm | none
ml.trend          หน่วย/วัน
ml.forecast24     พยากรณ์ 24 ชั่วโมง
ml.days_to_warn   อีกกี่วันถึงเกณฑ์เตือน
ml.days_to_alarm  อีกกี่วันถึงเกณฑ์ ALARM
ml.z_robust       ความเบี่ยงเบนแบบทนค่าโดด
ml.mae            ความคลาดเคลื่อนของโมเดลที่ใช้
ml.mae_baseline   ความคลาดเคลื่อนของ Holt-Winters ไว้เทียบ
ml.computedAt     เวลาที่คำนวณ
```

`index.html` ยังคำนวณคะแนนในเครื่องเอง ไม่ได้อ่านฟิลด์นี้ — ถ้าจะให้แสดงผลจาก LSTM ต้องแก้หน้าเว็บให้อ่าน `ml.score`

## ข้อควรรู้

- cron ของ GitHub อาจดีเลย์ 5–15 นาทีตอนคนใช้เยอะ ไม่สำคัญกับงานรายวัน
- โควตา repo ส่วนตัว 2,000 นาที/เดือน งานนี้ใช้ประมาณ 3–8 นาที/วัน
- `tensorflow-cpu` ขนาดใหญ่ ขั้นติดตั้งใช้เวลาประมาณ 1–2 นาที (มี pip cache ช่วย)
- ถ้าไม่ต้องการ LSTM เลย ลบ `tensorflow-cpu` ออกจาก requirements.txt ได้ สคริปต์จะใช้ Holt-Winters ล้วนโดยไม่ error
- สคริปต์เขียนเฉพาะฟิลด์ `ml` ไม่แตะ `value` / `bl` / `hourly` ที่ n8n และหน้าเว็บเป็นเจ้าของ
