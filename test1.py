  try:
        while True:
            # Receive frame as base64 string
            data = await websocket.receive_text()
            
            # Strip data URL prefix if present
            if "," in data:
                data = data.split(",")[1]
                
            # Decode base64 to numpy array for OpenCV
            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                boxes = []
                face_count = 0
                id_card_count = 0

                # 1. Face Detection Inference
                if face_model is not None:
                    face_results = face_model(img, conf=0.25, verbose=False)
                    for result in face_results:
                        for box in result.boxes:
                            b = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            label_name = face_model.names.get(cls_id, "FACE") if hasattr(face_model, "names") else "FACE"
                            boxes.append({
                                "x1": b[0],
                                "y1": b[1],
                                "x2": b[2],
                                "y2": b[3],
                                "confidence": conf,
                                "label": label_name,
                                "type": "face"
                            })
                            face_count += 1

                # 2. ID Card Detection Inference
                if id_card_model is not None:
                    id_results = id_card_model(img, conf=0.25, verbose=False)
                    for result in id_results:
                        for box in result.boxes:
                            b = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            label_name = id_card_model.names.get(cls_id, "ID Card") if hasattr(id_card_model, "names") else "ID Card"
                            boxes.append({
                                "x1": b[0],
                                "y1": b[1],
                                "x2": b[2],
                                "y2": b[3],
                                "confidence": conf,
                                "label": label_name,
                                "type": "id_card"
                            })
                            id_card_count += 1
                
                # Send combined results back to client
                await websocket.send_text(json.dumps({
                    "boxes": boxes,
                    "counts": {
                        "faces": face_count,
                        "id_cards": id_card_count
                    }
                }))
            else:
                await websocket.send_text(json.dumps({"error": "Failed to decode image"}))
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error processing frame: {e}")