Deploying and managing multiple instances of language models using Docker and FastAPI involves creating Docker containers for each instance, orchestrating these containers efficiently, and ensuring they are accessible through FastAPI endpoints. Here's a step-by-step guide to achieve this:

### Step 1: Prepare Your FastAPI Application

1. **FastAPI Application Setup**: Ensure your FastAPI application is structured to dynamically interact with different language model instances. Each instance could be running in its own Docker container.

2. **Endpoint Configuration**: Design FastAPI endpoints to route requests to specific language model instances based on the request parameters or headers.

### Step 2: Dockerize Your Language Model

1. **Dockerfile**: Create a Dockerfile for your language model application. This file should include instructions to set up the environment, install dependencies, and run the application.

2. **Build Docker Image**: Build a Docker image from your Dockerfile. This image will serve as the template for each language model instance.
   ```bash
   docker build -t lm-model:latest .
   ```

### Step 3: Deploy Multiple Instances

1. **Run Docker Containers**: Deploy multiple instances of your language model by running the Docker image multiple times, each time specifying different ports if necessary. For example:
   ```bash
   docker run -d --name lm-instance1 -p 8001:8000 lm-model:latest
   docker run -d --name lm-instance2 -p 8002:8000 lm-model:latest
   ```
   Adjust the `-p` flag to map the container's port to unique host ports if they need to be externally accessible.

### Step 4: Orchestrate with Docker Compose or Kubernetes

For managing multiple containers efficiently, consider using **Docker Compose** for a simple setup or **Kubernetes** for more complex deployments.

- **Docker Compose**: Create a `docker-compose.yml` file to define and run multi-container Docker applications. You can specify each language model instance as a service, managing them together.
  
- **Kubernetes**: For larger deployments, Kubernetes can help manage containerized applications across a cluster of machines. You can define deployments and services for your language model instances.

### Step 5: Load Balancing and Scaling

- **Nginx or Traefik**: Use Nginx or Traefik as a reverse proxy to distribute traffic among your instances and enable load balancing.

- **Autoscaling**: Kubernetes supports autoscaling based on traffic, ensuring that the number of instances scales up or down according to demand.

### Step 6: Monitoring and Management

- **Logging and Monitoring**: Implement logging and monitoring using tools like Prometheus and Grafana to keep track of the health and performance of your instances.

- **Management**: Use Kubernetes Dashboard or Portainer for Docker to manage and monitor your containers visually.

### Example FastAPI Route for Dynamic Instance Interaction

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/model/{model_id}")
async def interact_with_model(model_id: str, query: str):
    # Logic to route to the correct model instance based on model_id
    # This could involve sending a request to the appropriate Docker container
    # Example:
    # response = requests.post(f"http://localhost:{model_port_map[model_id]}/predict", json={"query": query})
    # return response.json()
    pass
```

This setup allows you to deploy and manage multiple instances of language models efficiently, scale according to demand, and maintain high availability and performance.


Configuring load balancing for multiple instances of language models deployed with Docker and FastAPI involves setting up a reverse proxy that distributes incoming requests among your instances. Nginx is a popular choice for this task due to its efficiency and flexibility. Here's how to set it up:

### Step 1: Deploy Your FastAPI Instances

1. **Deploy Multiple Instances**: Deploy multiple instances of your FastAPI application, each running in a Docker container. Ensure they are running on different ports if they are on the same host.
   ```bash
   docker run -d --name lm-instance1 -p 8001:8000 lm-model:latest
   docker run -d --name lm-instance2 -p 8002:8000 lm-model:latest
   ```

### Step 2: Set Up Nginx for Load Balancing

1. **Install Nginx**: If you haven't already, install Nginx on your host machine or within a Docker container.
   ```bash
   sudo apt update
   sudo apt install nginx -y
   ```

2. **Configure Nginx**: Create or modify an Nginx configuration file for load balancing. Replace `/etc/nginx/nginx.conf` or create a new file in `/etc/nginx/conf.d/` with the following configuration, adjusting the `server` entries under `upstream` to point to your FastAPI instances.
   ```nginx
   http {
       upstream lm_backend {
           server localhost:8001;
           server localhost:8002;
       }

       server {
           listen 80;

           location / {
               proxy_pass http://lm_backend;
               proxy_set_header Host $host;
               proxy_set_header X-Real-IP $remote_addr;
               proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
               proxy_set_header X-Forwarded-Proto $scheme;
           }
       }
   }
   ```

3. **Reload Nginx**: Apply the changes by reloading Nginx.
   ```bash
   sudo nginx -s reload
   ```

### Step 3: Verify Load Balancing

- **Testing**: Use `curl` or a web browser to access your Nginx server's IP address or domain name. Nginx will distribute the requests among your FastAPI instances.
  ```bash
  curl http://localhost/
  ```

### Additional Tips

- **Dockerize Nginx**: For a more scalable and isolated setup, consider running Nginx in a Docker container and use Docker networking features to connect it to your FastAPI containers.

- **SSL/TLS**: Secure your setup with SSL/TLS by obtaining certificates (e.g., from Let's Encrypt) and configuring Nginx to use HTTPS.

- **Advanced Load Balancing**: Explore Nginx's advanced load balancing features, such as IP hash or least connections, to further optimize the distribution of requests.

- **Health Checks**: Implement health checks in Nginx to ensure traffic is only routed to healthy instances of your application.

This setup provides a basic yet robust approach to load balancing for FastAPI applications running in Docker containers, ensuring efficient distribution of requests and increased availability.

Monitoring the performance and resource usage of your FastAPI application running in Docker containers can be achieved through a combination of Docker-native tools and third-party monitoring solutions. Here's a comprehensive approach:

### Step 1: Use Docker's Built-in Commands

1. **Docker Stats**: The `docker stats` command provides a live stream of container(s) resource usage statistics such as CPU, memory usage, I/O, and network metrics.
   ```bash
   docker stats
   ```

2. **Docker Logs**: For performance insights related to application behavior, use `docker logs` to access the logs of a running container.
   ```bash
   docker logs <container_name_or_id>
   ```

### Step 2: Implement Application-Level Metrics in FastAPI

1. **FastAPI Middleware for Metrics**: Implement middleware in your FastAPI application to collect request timings, error rates, and other relevant metrics. You can use libraries like `starlette-prometheus` to easily expose these metrics.

2. **Prometheus Integration**:
   - Integrate Prometheus with your FastAPI application to scrape and store the metrics exposed by your middleware.
   - Configure Prometheus to scrape metrics at a regular interval.

### Step 3: Visualize Metrics with Grafana

1. **Set Up Grafana**: Install and configure Grafana to visualize the metrics collected by Prometheus.
   - Connect Grafana to your Prometheus instance as a data source.
   - Create dashboards in Grafana to visualize the metrics, such as request rates, error rates, response times, and system resource usage.

### Step 4: Use cAdvisor for Container Metrics

1. **Deploy cAdvisor**: cAdvisor (Container Advisor) provides container users an understanding of the resource usage and performance characteristics of their running containers.
   - Run cAdvisor in a Docker container to monitor all other containers:
     ```bash
     docker run \
       --volume=/:/rootfs:ro \
       --volume=/var/run:/var/run:rw \
       --volume=/sys:/sys:ro \
       --volume=/var/lib/docker/:/var/lib/docker:ro \
       --publish=8080:8080 \
       --detach=true \
       --name=cadvisor \
       google/cadvisor:latest
     ```

### Step 5: Aggregate Logs with ELK Stack or Fluentd

1. **ELK Stack (Elasticsearch, Logstash, Kibana)**: Use the ELK stack for log aggregation and visualization.
   - **Elasticsearch** stores and indexes logs.
   - **Logstash** processes and sends logs to Elasticsearch.
   - **Kibana** visualizes the logs stored in Elasticsearch.

2. **Fluentd**: Alternatively, use Fluentd as a log aggregator, which can collect logs from Docker containers and forward them to various destinations such as Elasticsearch for storage and analysis.

### Step 6: Set Up Alerts

1. **Alerting with Prometheus and Grafana**: Configure alert rules in Prometheus based on your metrics. Use Grafana to send notifications via email, Slack, or other channels when metrics breach your defined thresholds.

### Example: FastAPI Middleware for Prometheus Metrics

```python
from fastapi import FastAPI, Request
from starlette_prometheus import metrics, PrometheusMiddleware

app = FastAPI()

# Add Prometheus middleware to collect metrics
app.add_middleware(PrometheusMiddleware)
# Expose metrics endpoint
app.add_route("/metrics/", metrics)

@app.get("/")
async def main():
    return {"message": "Hello World"}
```

This setup provides a comprehensive monitoring solution, covering live resource usage, application performance metrics, log aggregation, visualization, and alerting. It ensures you have the insights needed to maintain and optimize your FastAPI application running in Docker containers.

Optimizing resource allocation for a FastAPI application running in Docker containers involves several strategies to ensure efficient use of system resources while maintaining performance. Here's a step-by-step guide:

### Step 1: Analyze Current Resource Usage

Before making adjustments, understand your application's current resource usage to identify bottlenecks or over-allocations.

- Use `docker stats` to monitor CPU and memory usage in real-time.
- Employ profiling tools within your FastAPI application to identify slow functions or endpoints.

### Step 2: Set Resource Limits in Docker

Limiting resources per container can prevent any single application from consuming all available resources, ensuring stability across services.

```dockerfile
# Example Dockerfile snippet setting resource limits
FROM python:3.8-slim
# Your application setup here

# Example docker-compose.yml snippet setting resource limits
services:
  fastapi-app:
    build: .
    ports:
      - "80:80"
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 500M
        reservations:
          cpus: '0.25'
          memory: 200M
```

- **CPUs**: Limit the CPU usage with the `--cpus` flag when running a container or in your `docker-compose.yml`.
- **Memory**: Set memory limits using the `--memory` flag or in your `docker-compose.yml`.

### Step 3: Optimize Docker Images

- **Use Slimmer Base Images**: Choose a slimmer base image if possible, such as `python:3.8-slim`, to reduce the overall footprint of your Docker images.
- **Multi-Stage Builds**: Use multi-stage builds in your Dockerfile to minimize the size of the final image by separating the build environment from the runtime environment.

### Step 4: Implement Efficient Coding Practices

- **Asynchronous Programming**: Leverage FastAPI's support for asynchronous request handling to improve the efficiency of I/O-bound operations.
- **Dependency Optimization**: Regularly review and optimize your application's dependencies, removing unnecessary packages.

### Step 5: Enable Auto-Scaling

If you're deploying your containers in a cluster managed by an orchestrator like Kubernetes, take advantage of auto-scaling features to dynamically adjust the number of container instances based on load.

- **Horizontal Pod Autoscaler (HPA)** in Kubernetes can automatically scale the number of pods in a deployment based on observed CPU utilization or other selected metrics.

### Step 6: Use Efficient Data Storage and Retrieval

- **Database Connections**: Use connection pooling to manage database connections efficiently.
- **Caching**: Implement caching strategies for frequently accessed data to reduce database load and improve response times.

### Step 7: Monitor and Adjust

- Continuously monitor your application's performance and resource usage.
- Adjust resource limits and application optimizations based on observed performance and changing requirements.

### Example: Docker Compose with Resource Limits

```yaml
version: '3.8'
services:
  fastapi-app:
    build: .
    ports:
      - "80:80"
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 500M
        reservations:
          cpus: '0.25'
          memory: 200M
```

By following these steps, you can optimize the resource allocation for your FastAPI application running in Docker containers, ensuring efficient use of resources without sacrificing performance.

Scaling your FastAPI application running in Docker containers horizontally involves increasing the number of container instances to handle more load. This can be achieved manually or automatically, depending on your infrastructure setup. Here's a step-by-step guide focusing on Docker Swarm and Kubernetes, two popular container orchestration tools that support automatic scaling.

### Using Docker Swarm

1. **Initialize Docker Swarm**: If not already initialized, turn your Docker engine into a Swarm manager node.
   ```bash
   docker swarm init
   ```

2. **Deploy Your FastAPI Application as a Service**: Deploy your application as a service in Docker Swarm, which allows you to scale it easily.
   ```bash
   docker service create --name fastapi-app --publish 80:80 --replicas 3 your_fastapi_image
   ```

3. **Scale the Service**: Adjust the number of replicas for your service based on the load.
   ```bash
   docker service scale fastapi-app=5
   ```

### Using Kubernetes

1. **Deploy Your Application on Kubernetes**: Create a deployment in Kubernetes for your FastAPI application. Ensure you have a Kubernetes cluster running and `kubectl` configured.
   ```yaml
   # fastapi-deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: fastapi-app
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: fastapi
     template:
       metadata:
         labels:
           app: fastapi
       spec:
         containers:
         - name: fastapi
           image: your_fastapi_image
           ports:
           - containerPort: 80
   ```

   Apply the deployment:
   ```bash
   kubectl apply -f fastapi-deployment.yaml
   ```

2. **Expose Your Application**: Expose your FastAPI application to the internet using a Kubernetes service.
   ```yaml
   # fastapi-service.yaml
   kind: Service
   apiVersion: v1
   metadata:
     name: fastapi-service
   spec:
     selector:
       app: fastapi
     ports:
     - protocol: TCP
       port: 80
       targetPort: 80
     type: LoadBalancer
   ```

   Apply the service:
   ```bash
   kubectl apply -f fastapi-service.yaml
   ```

3. **Implement Auto-Scaling**: Use the Horizontal Pod Autoscaler (HPA) to automatically scale your application based on CPU or memory usage.
   ```bash
   kubectl autoscale deployment fastapi-app --cpu-percent=50 --min=3 --max=10
   ```

### Monitoring and Adjusting

- **Monitor Performance**: Use tools like Prometheus and Grafana to monitor the performance of your FastAPI application. Adjust the scaling parameters based on the observed load and performance metrics.

- **Load Balancing**: Ensure your load balancer (e.g., Nginx, HAProxy, or cloud provider load balancers) is properly configured to distribute traffic evenly across all instances of your application.

- **State Management**: For stateful applications, consider how to manage session state across instances. Options include using external session stores like Redis or sticky sessions at the load balancer level.

By following these steps, you can scale your FastAPI application horizontally to handle increased load, improving availability and responsiveness.

Implementing authentication and authorization in a FastAPI application typically involves several steps, including setting up user models, creating authentication endpoints, and securing routes with dependencies. Below is a step-by-step guide to implement a basic JWT (JSON Web Tokens) based authentication system, which is a common approach for web and API services.

### Step 1: Install Dependencies

First, install the necessary libraries: `fastapi`, `pydantic`, `passlib`, and `python-jose`.

```bash
pip install fastapi uvicorn pydantic passlib[bcrypt] python-jose[cryptography]
```

### Step 2: Create User Models

Define Pydantic models for user data and authentication payloads.

```python
from pydantic import BaseModel

class User(BaseModel):
    username: str
    email: str
    full_name: str = None
    disabled: bool = None

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str = None
```

### Step 3: Password Utility Functions

Create utility functions for hashing and verifying passwords.

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)
```

### Step 4: Create JWT Utility Functions

Implement functions to generate and verify JWT tokens.

```python
from datetime import datetime, timedelta
from jose

 import

 JWTError, jwt

SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    return token_data
```

### Step 5: Authentication Endpoint

Create an endpoint for user authentication that issues JWT tokens upon successful login.

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
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

### Step 6: Secure Routes with Dependencies

Use the `Depends` function to secure your API routes, requiring a valid JWT token for access.

```python
from fastapi import Security

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    return verify_token(token, credentials_exception)

@app.get("/users/me")
async def read_users_me(current_user: User = Security(get_current_user)):
    return current_user
```

### Step 7: Test Your Authentication

Run your FastAPI application and test the authentication flow using tools like `curl` or Postman. Ensure that protected routes are inaccessible without a valid token and that the login endpoint correctly issues tokens.

This guide provides a basic framework for authentication and authorization in FastAPI using JWT. Depending on your requirements, you might need to adjust user models, token management, and security measures.

Handling user registration and account management in a FastAPI application involves creating endpoints for user registration, user data management, and possibly password reset functionality. Below is a simplified approach to implement these features.

### Step 1: Setup Database Models

First, define your user model for the database. This example uses SQLAlchemy for ORM, but you can adapt it to your preferred database library.

```python
from sqlalchemy import Column, Integer, String, Boolean
from database import Base  # Ensure you have a Base class from SQLAlchemy ORM

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
```

### Step 2: Create Pydantic Schemas

Define Pydantic schemas for user registration and user data.

```python
from pydantic import BaseModel

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool

    class Config:
        orm_mode = True
```

### Step 3: User Registration Endpoint

Implement an endpoint for user registration. This includes hashing the password before storing it in the database.

```python
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from models import User  # Your SQLAlchemy user model
from schemas import UserCreate  # Your Pydantic user creation schema
from database import SessionLocal, engine  # Your database session and engine

app = FastAPI()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users/", response_model=schemas.User)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = pwd_context.hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password, username=user.username)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
```

### Step 4: Account Management Endpoints

Add endpoints for viewing and updating user information. Ensure you have proper authentication in place to protect these endpoints.

```python
@app.get("/users/{user_id}", response_model=User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.put("/users/{user_id}", response_model=User)
def update_user(user_id: int, user: UserUpdate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    user_data = user.dict(exclude_unset=True)
    for key, value in user_data.items():
        setattr(db_user, key, value)
    db.commit()
    db.refresh(db_user)
    return db_user
```

### Step 5: Password Reset (Optional)

Implementing a password reset feature typically involves sending a reset link to the user's email, which requires integrating an email service into your application.

This guide provides a basic framework for user registration and account management in FastAPI. Depending on your application's requirements, you might need to add more features like email verification, password strength validation, and more sophisticated user management capabilities.

Implementing Role-Based Access Control (RBAC) in a FastAPI application involves defining roles, associating them with users, and then using those roles to control access to various parts of your application. Here's a step-by-step guide to implement a basic RBAC system:

### Step 1: Define User and Role Models

First, define your user and role models, including a many-to-many relationship between them if you're using an ORM like SQLAlchemy.

```python
from sqlalchemy import Table, Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from database import Base

# Association table for the many-to-many relationship
roles_users = Table('roles_users', Base.metadata,
    Column('user_id', Integer, ForeignKey('user.id')),
    Column('role_id', Integer, ForeignKey('role.id'))
)

class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    roles = relationship("Role", secondary=roles_users, back_populates="users")

class Role(Base):
    __tablename__ = 'role'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    users = relationship("User", secondary=roles_users, back_populates="roles")
```

### Step 2: Create Roles and Assign Them to Users

After defining the models, create roles in your database and assign them to users. This could be done through a registration process, an admin interface, or directly in the database.

### Step 3: Implement Authentication and Role Check

Implement authentication (e.g., with JWT tokens as described in a previous example) and add a dependency to check a user's roles.

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from . import crud, models
from .database import SessionLocal

oauth2

_scheme

 = OAuth2PasswordBearer(tokenUrl="token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
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
    user = crud.get_user_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

def get_current_active_user(current_user: models.User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def has_role(role: str):
    def role_checker(current_user: models.User = Depends(get_current_active_user)):
        if not any(r.name == role for r in current_user.roles):
            raise HTTPException(status_code=403, detail="Operation not permitted")
        return current_user
    return role_checker
```

### Step 4: Secure Endpoints with Role Checks

Use the `has_role` dependency to secure your endpoints, allowing only users with the required roles to access them.

```python
from fastapi import FastAPI, Depends

app = FastAPI()

@app.get("/items/", dependencies=[Depends(has_role("admin"))])
async def read_items():
    return [{"item_id": "Foo"}, {"item_id": "Bar"}]
```

### Step 5: Test Your RBAC Implementation

Test your endpoints with users having different roles to ensure that access control is working as expected.

This guide provides a basic framework for implementing RBAC in FastAPI. Depending on your application's complexity, you might need to extend this system with more features, such as role hierarchies or permission-based access control.
