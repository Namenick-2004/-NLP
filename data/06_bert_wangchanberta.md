# BERT และ WangchanBERTa

BERT ย่อมาจาก Bidirectional Encoder Representations from Transformers พัฒนาโดย Google และเผยแพร่ในปี 2018 เป็นโมเดลที่ใช้เฉพาะ Encoder ของ Transformer ลักษณะเด่นคือการเรียนรู้บริบทแบบสองทิศทาง (Bidirectional) คือมองทั้งคำก่อนหน้าและคำถัดไปพร้อมกัน

## การฝึกล่วงหน้า (Pre-training) ของ BERT

BERT ฝึกล่วงหน้าด้วยงานสองแบบ แบบแรกคือ Masked Language Modeling (MLM) สุ่มปิดบังคำในประโยคประมาณ 15% แล้วให้โมเดลทำนายคำที่ถูกปิด แบบที่สองคือ Next Sentence Prediction (NSP) ให้ทำนายว่าประโยคสองประโยคต่อเนื่องกันจริงหรือไม่ หลังฝึกล่วงหน้าแล้ว สามารถนำไปปรับจูน (Fine-tuning) กับงานเฉพาะ เช่น การจำแนกข้อความ โดยเพิ่มชั้นสำหรับการจำแนกต่อท้ายและฝึกต่อด้วยข้อมูลที่มีป้ายกำกับจำนวนไม่มาก

## โทเคนพิเศษของ BERT

โทเคน [CLS] ถูกเพิ่มไว้ต้นประโยคเพื่อใช้เป็นตัวแทนของทั้งประโยคในงานจำแนก ส่วน [SEP] ใช้คั่นระหว่างประโยค และ [MASK] ใช้แทนคำที่ถูกปิดบังตอนฝึก MLM

## WangchanBERTa

WangchanBERTa เป็นโมเดลภาษาไทยที่ฝึกล่วงหน้าโดยสถาบันวิจัยปัญญาประดิษฐ์ VISTEC ร่วมกับ AI Research Institute of Thailand (AIResearch.in.th) และ PyThaiNLP ใช้สถาปัตยกรรม RoBERTa ซึ่งเป็นการปรับปรุงวิธีฝึกของ BERT โดยตัดงาน NSP ออกและฝึกด้วยข้อมูลมากขึ้น โมเดลฝึกด้วยข้อความภาษาไทยขนาดประมาณ 78 GB จากแหล่งข้อมูลหลากหลาย เช่น ข่าว วิกิพีเดีย และโซเชียลมีเดีย

ชื่อโมเดลบน Hugging Face คือ airesearch/wangchanberta-base-att-spm-uncased ซึ่งใช้ SentencePiece ในการตัดโทเคน และ airesearch/wangchanberta-base-wiki-newmm ซึ่งใช้การตัดคำ newmm งานที่นำ WangchanBERTa ไปปรับจูนได้ผลดี เช่น การวิเคราะห์ความรู้สึก (เช่นชุดข้อมูล wisesight_sentiment) การจำแนกหมวดหมู่ และการรู้จำชื่อเฉพาะ

## ตัวอย่างการใช้งานผ่าน Hugging Face

ใช้คลาส AutoTokenizer และ AutoModelForSequenceClassification จากไลบรารี transformers โหลดโมเดลด้วยชื่อ airesearch/wangchanberta-base-att-spm-uncased กำหนดจำนวนป้ายกำกับ (num_labels) แล้วฝึกต่อด้วย Trainer API
