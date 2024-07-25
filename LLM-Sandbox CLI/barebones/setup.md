For the MVP of the 'lm-sandbox-cli' project, we want to achieve the necessary functionality where local language models (LLMs) run on local hardware and can access and operate within a VM Ubuntu Bash CLI environment. Here's a step-by-step guide to achieve this:

### Ground Game for MVP

1. **Set Up Project Structure**
2. **Implement Basic FastAPI Server**
3. **Dockerize the Application**
4. **Integrate Language Models**
5. **Create Endpoints for LLM Interaction and Bash Command Execution**
6. **Set Up Authentication**
7. **Testing and Validation**

### 1. Set Up Project Structure

Create a project directory structure as follows:

```plaintext
lm-sandbox-cli/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── llm.py
│   │   ├── bash.py
│   │   └── auth.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py
│   │   ├── bash_service.py
│   │   └── auth_service.py
│   └── utils/
│       ├── __init__.py
│       └── logging.py
│
├── tests/
│   ├── __init__.py
│   ├── test_llm.py
│   ├── test_bash.py
│   └── test_auth.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .env
```

### 2. Implement Basic FastAPI Server

#### app/main.py

```python
from fastapi import FastAPI
from app.routers import llm, bash, auth

app = FastAPI()

app.include_router(auth.router)
app.include_router(llm.router)
app.include_router(bash.router)
```

#### app/models.py

```python
from pydantic import BaseModel

class User(BaseModel):
    username: str
    full_name: str
    email: str
    disabled: bool = False

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class LLMInput(BaseModel):
    input_text: str

class BashInput(BaseModel):
    command: str
```

### 3. Dockerize the Application

#### Dockerfile

```Dockerfile
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Install necessary packages
RUN apt-get update && apt-get install -y bash

# Copy application code
COPY ./app /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 80

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  fastapi:
    build: .
    ports:
      - "8000:80"
    environment:
      - ENV=development
    volumes:
      - ./app:/app
```

### 4. Integrate Language Models

#### app/services/llm_service.py

```python
import re

def process_input(input_text: str) -> str:
    # Mocked response from the language model
    response = f"Processed text: {input_text}"
    
    # Example parsing logic
    parsed_response = re.sub(r'\s+', ' ', response)
    return parsed_response
```

#### app/services/bash_service.py

```python
import subprocess

def execute_command(command: str) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(result.stderr)
    return result.stdout
```

### 5. Create Endpoints for LLM Interaction and Bash Command Execution

#### app/routers/llm.py

```python
from fastapi import APIRouter, Depends, HTTPException
from app.services.llm_service import process_input
from app.models import LLMInput
from app.routers.auth import get_current_user

router = APIRouter(
    prefix="/llm",
    tags=["llm"]
)

@router.post("/process")
async def process_llm_input(input_data: LLMInput, current_user: dict = Depends(get_current_user)):
    try:
        result = process_input(input_data.input_text)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### app/routers/bash.py

```python
from fastapi import APIRouter, Depends, HTTPException
from app.services.bash_service import execute_command
from app.models import BashInput
from app.routers.auth import get_current_user

router = APIRouter(
    prefix="/bash",
    tags=["bash"]
)

@router.post("/execute")
async def execute_bash_command(input_data: BashInput, current_user: dict = Depends(get_current_user)):
    try:
        result = execute_command(input_data.command)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 6. Set Up Authentication

#### app/routers/auth.py

```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from app.models import User, UserInDB, Token, TokenData

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

fake_users_db = {
    "user": {
        "username": "user",
        "full_name": "User Fullname",
        "email": "user@example.com",
        "hashed_password": "fakehashedpassword",
        "disabled": False,
    }
}

def verify_password(plain_password, hashed_password):
    return plain_password == hashed_password

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
```

### 7. Testing and Validation

#### tests/test_llm.py

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_process_llm_input():
    response = client.post("/llm/process", json={"input_text": "Test input"})
    assert response.status_code == 200
    assert "result" in response.json()
    assert response.json()["result"] == "Processed text: Test input"
```

#### tests/test_bash.py

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_execute_bash_command():
    response = client.post("/bash/execute", json={"command": "echo Hello World"})
    assert response.status_code == 200
    assert "result" in response.json()
    assert response.json()["result"].strip() == "Hello World"
```

#### tests/test_auth.py

```python
from fastapi.testclient import TestClient
from

 app.main import app

client = TestClient(app)

def test_login_for_access_token():
    response = client.post("/token", data={"username": "user", "password": "password"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"
```

### Running the Application

1. **Build and Run the Application**:

```bash
docker-compose up --build
```

2. **Access the API**:

- Open your browser or use a tool like Postman to interact with the API at `http://localhost:8000`.

3. **Run Tests**:

```bash
pytest
```

### Final Thoughts

This setup provides a functional MVP for the `lm-sandbox-cli` project, covering core features such as language model interaction, bash command execution, authentication, and testing. This foundation can be expanded and enhanced with more sophisticated features and optimizations as needed.
