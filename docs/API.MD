## Kissan CV Service API Documentation

### **Base URL**
All endpoints are prefixed with `/api/v1`.

---

### **Scan & Analysis Endpoints**

#### **Crop Scan**
- **URL**: `/crop_scan`
- **Method**: `POST`
- **Summary**: upload crop image to detect disease
- **Request Body**: `multipart/form-data` with `file` (image) and `user_id`.
- **Response**: JSON containing disease analysis, severity, and treatment.

#### **Fruit Scan**
- **URL**: `/fruit_scan`
- **Method**: `POST`
- **Summary**: upload fruit image for analysis
- **Request Body**: `multipart/form-data` with `file` (image) and `user_id`.

#### **Vegetable Scan**
- **URL**: `/vegetable_scan`
- **Method**: `POST`
- **Summary**: upload vegetable image for analysis
- **Request Body**: `multipart/form-data` with `file` (image) and `user_id`.

#### **Dynamic Scan**
- **URL**: `/dynamic_scan`
- **Method**: `POST`
- **Summary**: General purpose scan endpoint.

#### **Generate Report**
- **URL**: `/report/generate`
- **Method**: `POST`
- **Summary**: Generates a detailed PDF report for a specific analysis ID.

---

### **Payment Endpoints**

#### **Create Order**
- **URL**: `/payments/create-order`
- **Method**: `POST`
- **Summary**: Creates a payment order for a report.
- **Request Body**:
  ```json
  {
    "amount": 1.0,
    "currency": "INR",
    "user_id": "string",
    "report_id": "string",
    "category": "string"
  }
  ```
- **Response**: Returns `order_id` and payment links.

#### **Verify Payment**
- **URL**: `/payments/verify/{order_id}`
- **Method**: `POST`
- **Summary**: Verifies the status of a payment.
- **Response**: Returns payment status (SUCCESS/PENDING/FAILED).

#### **Payment Webhook**
- **URL**: `/payments/webhook`
- **Method**: `POST`
- **Summary**: Webhook for payment gateway callbacks.

---

### **Translation Endpoints**

#### **Translate Report**
- **URL**: `/translate/report`
- **Method**: `POST`
- **Summary**: Translates an existing analysis report to a target language (e.g., Hindi, Kannada).

---

### **Async Scan Endpoints**

#### **Task Status**
- **URL**: `/task/{task_id}`
- **Method**: `GET`
- **Summary**: Check status of an asynchronous scan task.

