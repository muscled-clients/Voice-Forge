# VoiceForge GraphQL API Documentation

VoiceForge provides a comprehensive GraphQL API alongside the REST API, offering flexible data fetching and real-time subscriptions for speech-to-text operations.

## Overview

The GraphQL API is available at `/graphql` and provides:

- **Queries**: Fetch user data, transcription jobs, analytics, and system information
- **Mutations**: Create accounts, manage transcriptions, and update user profiles  
- **Subscriptions**: Real-time updates for transcription progress and system status

## Getting Started

### GraphQL Playground

Visit `http://localhost:8000/graphql` to access the interactive GraphQL playground where you can:

- Explore the schema documentation
- Write and test queries interactively
- View real-time subscription updates

### Authentication

Include your access token in the Authorization header:

```
Authorization: Bearer your_access_token_here
```

## Schema Overview

### Core Types

```graphql
type User {
  id: String!
  email: String!
  fullName: String!
  tier: UserTier!
  creditsRemaining: Int!
  isActive: Boolean!
  createdAt: DateTime!
  updatedAt: DateTime
}

type TranscriptionJob {
  id: String!
  userId: String!
  status: TranscriptionStatus!
  createdAt: DateTime!
  updatedAt: DateTime
  completedAt: DateTime
  
  # Input details
  filename: String!
  audioFormat: AudioFormat
  audioDuration: Float
  audioSize: Int
  
  # Processing details
  modelUsed: String
  languageCode: String
  processingTime: Float
  
  # Results
  transcript: String
  confidence: Float
  words: [Word!]
  diarization: [DiarizationSegment!]
  detectedLanguage: LanguageDetectionResult
  
  # Metadata
  wordCount: Int
  metadata: JSON
  
  # Error info
  error: String
  errorDetails: JSON
}

type ModelInfo {
  name: String!
  modelId: String!
  type: String!
  description: String!
  supportedLanguages: [String!]!
  maxDuration: Int
  accuracyScore: Float
  latencyMs: Int
  isAvailable: Boolean!
  requiresGpu: Boolean!
}
```

### Enums

```graphql
enum TranscriptionStatus {
  PENDING
  PROCESSING
  COMPLETED
  FAILED
  CANCELLED
}

enum UserTier {
  FREE
  BASIC
  PRO
  ENTERPRISE
}

enum AudioFormat {
  WAV
  MP3
  M4A
  FLAC
  OGG
  WEBM
}
```

## Queries

### Get Current User

```graphql
query {
  currentUser {
    id
    email
    fullName
    tier
    creditsRemaining
    isActive
    createdAt
  }
}
```

### List Transcription Jobs

```graphql
query GetTranscriptionJobs($pagination: PaginationInput, $status: TranscriptionStatus) {
  transcriptionJobs(pagination: $pagination, status: $status) {
    jobs {
      id
      status
      filename
      createdAt
      completedAt
      transcript
      confidence
      processingTime
    }
    totalCount
    page
    totalPages
    hasNext
    hasPrev
  }
}
```

**Variables:**
```json
{
  "pagination": {
    "page": 1,
    "perPage": 20,
    "sortBy": "createdAt",
    "sortOrder": "desc"
  },
  "status": "COMPLETED"
}
```

### Get Specific Transcription Job

```graphql
query GetTranscriptionJob($jobId: String!) {
  transcriptionJob(jobId: $jobId) {
    id
    status
    filename
    transcript
    confidence
    words {
      text
      startTime
      endTime
      confidence
      speakerId
    }
    diarization {
      speakerId
      startTime
      endTime
      confidence
      text
    }
    detectedLanguage {
      code
      name
      confidence
      method
      alternatives {
        code
        name
        confidence
      }
    }
    processingTime
    createdAt
    completedAt
  }
}
```

### Get Available Models

```graphql
query {
  availableModels {
    name
    modelId
    type
    description
    supportedLanguages
    maxDuration
    accuracyScore
    latencyMs
    isAvailable
    requiresGpu
  }
}
```

### User Analytics

```graphql
query GetUserAnalytics($filter: AnalyticsFilter) {
  userAnalytics(filter: $filter) {
    userId
    totalTranscriptions
    totalAudioMinutes
    totalCreditsUsed
    averageConfidence
    mostUsedLanguage
    mostUsedModel
    successRate
    periodStart
    periodEnd
  }
}
```

**Variables:**
```json
{
  "filter": {
    "startDate": "2024-01-01T00:00:00Z",
    "endDate": "2024-01-31T23:59:59Z",
    "language": "en",
    "status": "COMPLETED"
  }
}
```

### System Metrics (Admin Only)

```graphql
query {
  systemMetrics {
    totalTranscriptions
    totalUsers
    activeJobs
    queueLength
    averageLatencyMs
    successRate
    supportedLanguages
    availableModels
    systemLoad
    timestamp
  }
}
```

### User Quota Information

```graphql
query {
  quotaInfo {
    userId
    tier
    creditsTotal
    creditsUsed
    creditsRemaining
    monthlyLimit
    monthlyUsed
    resetDate
  }
}
```

## Mutations

### User Registration

```graphql
mutation RegisterUser($input: UserRegistrationInput!) {
  registerUser(input: $input) {
    ... on AuthenticationResult {
      success
      message
      accessToken
      refreshToken
      user {
        id
        email
        fullName
        tier
      }
      expiresIn
    }
    ... on ErrorResult {
      success
      message
      code
      details
      validationErrors {
        field
        message
        code
      }
    }
  }
}
```

**Variables:**
```json
{
  "input": {
    "email": "user@example.com",
    "password": "securepassword123",
    "fullName": "John Doe",
    "company": "Example Corp"
  }
}
```

### User Login

```graphql
mutation LoginUser($input: UserLoginInput!) {
  loginUser(input: $input) {
    ... on AuthenticationResult {
      success
      message
      accessToken
      user {
        id
        email
        fullName
        tier
        creditsRemaining
      }
      expiresIn
    }
    ... on ErrorResult {
      success
      message
      code
    }
  }
}
```

**Variables:**
```json
{
  "input": {
    "email": "user@example.com",
    "password": "securepassword123"
  }
}
```

### Create Transcription Job

```graphql
mutation CreateTranscriptionJob($audioFile: Upload!, $input: TranscriptionInput) {
  createTranscriptionJob(audioFile: $audioFile, input: $input) {
    ... on TranscriptionResult {
      success
      message
      job {
        id
        status
        filename
        modelUsed
        languageCode
      }
      estimatedCompletion
    }
    ... on ErrorResult {
      success
      message
      code
      details
    }
  }
}
```

**Variables:**
```json
{
  "input": {
    "filename": "meeting-recording.wav",
    "audioFormat": "WAV",
    "options": {
      "model": "whisper-medium",
      "language": "en",
      "enableDiarization": true,
      "enablePunctuation": true,
      "enableWordTimestamps": true,
      "enableLanguageDetection": false,
      "maxSpeakers": 5,
      "boostKeywords": ["meeting", "presentation", "discussion"],
      "customVocabulary": ["VoiceForge", "API", "transcription"]
    },
    "metadata": {
      "project": "Q1 Meeting Notes",
      "department": "Engineering"
    }
  }
}
```

### Batch Transcription

```graphql
mutation CreateBatchTranscription($audioFiles: [Upload!]!, $input: BatchTranscriptionInput) {
  createBatchTranscription(audioFiles: $audioFiles, input: $input) {
    ... on BatchTranscriptionResult {
      success
      message
      jobs {
        id
        filename
        status
      }
      batchId
    }
    ... on ErrorResult {
      success
      message
      code
    }
  }
}
```

### Cancel Transcription Job

```graphql
mutation CancelTranscriptionJob($jobId: String!) {
  cancelTranscriptionJob(jobId: $jobId) {
    ... on TranscriptionResult {
      success
      message
      job {
        id
        status
      }
    }
    ... on ErrorResult {
      success
      message
      code
    }
  }
}
```

### Update User Profile

```graphql
mutation UpdateUserProfile($input: UserUpdateInput!) {
  updateUserProfile(input: $input) {
    ... on AuthenticationResult {
      success
      message
      user {
        id
        fullName
        company
        updatedAt
      }
    }
    ... on ErrorResult {
      success
      message
      code
    }
  }
}
```

## Subscriptions

### Transcription Updates

Subscribe to real-time updates for transcription jobs:

```graphql
subscription TranscriptionUpdates($jobId: String) {
  transcriptionUpdates(jobId: $jobId) {
    jobId
    status
    progress
    partialResult {
      transcript
      isFinal
      confidence
      timestamp
    }
    timestamp
  }
}
```

### Streaming Transcription

Real-time streaming transcription results:

```graphql
subscription StreamingTranscription($sessionId: String!) {
  streamingTranscription(sessionId: $sessionId)
}
```

### System Status (Admin Only)

Monitor system-wide status:

```graphql
subscription SystemStatus {
  systemStatus {
    queueLength
    activeJobs
    systemLoad
    timestamp
  }
}
```

### User Notifications

Receive user-specific notifications:

```graphql
subscription UserNotifications {
  userNotifications
}
```

## Error Handling

GraphQL responses use union types to handle success and error cases:

```graphql
# Successful response
{
  "data": {
    "createTranscriptionJob": {
      "success": true,
      "message": "Transcription job created successfully",
      "job": {
        "id": "job_123",
        "status": "PENDING"
      }
    }
  }
}

# Error response  
{
  "data": {
    "createTranscriptionJob": {
      "success": false,
      "message": "Insufficient credits",
      "code": "INSUFFICIENT_CREDITS",
      "details": {
        "required": 1,
        "available": 0
      }
    }
  }
}

# GraphQL syntax error
{
  "errors": [
    {
      "message": "Syntax Error: Expected Name, found }",
      "locations": [{"line": 3, "column": 5}]
    }
  ]
}
```

## Rate Limiting

The GraphQL API is subject to the same rate limiting as the REST API:

- **Free tier**: 60 requests per minute
- **Basic tier**: 300 requests per minute  
- **Pro tier**: 1000 requests per minute
- **Enterprise tier**: 5000 requests per minute

Rate limit information is included in response headers:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## Best Practices

### 1. Request Only Needed Fields

Take advantage of GraphQL's field selection:

```graphql
# Good - only request needed fields
query {
  transcriptionJobs {
    jobs {
      id
      status
      filename
    }
  }
}

# Avoid - requesting unnecessary data
query {
  transcriptionJobs {
    jobs {
      id
      status
      filename
      transcript  # Not needed if just showing job list
      words       # Heavy field, only request if needed
      diarization # Heavy field, only request if needed
    }
  }
}
```

### 2. Use Fragments for Reusability

```graphql
fragment JobSummary on TranscriptionJob {
  id
  status
  filename
  createdAt
  completedAt
  processingTime
}

query {
  transcriptionJobs {
    jobs {
      ...JobSummary
    }
  }
}

query($jobId: String!) {
  transcriptionJob(jobId: $jobId) {
    ...JobSummary
    transcript
    confidence
    words {
      text
      startTime
      endTime
    }
  }
}
```

### 3. Handle Pagination Properly

```graphql
query GetTranscriptionJobs($page: Int!, $perPage: Int!) {
  transcriptionJobs(
    pagination: { 
      page: $page, 
      perPage: $perPage,
      sortBy: "createdAt",
      sortOrder: "desc"
    }
  ) {
    jobs {
      id
      status
      filename
      createdAt
    }
    totalCount
    page
    totalPages
    hasNext
    hasPrev
  }
}
```

### 4. Use Variables for Dynamic Queries

```graphql
# Good - using variables
query GetUserJobs($userId: String!, $status: TranscriptionStatus) {
  transcriptionJobs(status: $status) {
    jobs {
      id
      status
      filename
    }
  }
}

# Avoid - hardcoded values
query {
  transcriptionJobs(status: COMPLETED) {
    jobs {
      id
      status
      filename
    }
  }
}
```

### 5. Handle Subscriptions Properly

```javascript
// Client-side subscription handling
const subscription = client.subscribe({
  query: gql`
    subscription TranscriptionUpdates($jobId: String!) {
      transcriptionUpdates(jobId: $jobId) {
        jobId
        status
        progress
        timestamp
      }
    }
  `,
  variables: { jobId: 'job_123' }
});

subscription.subscribe({
  next: (data) => {
    console.log('Transcription update:', data);
    // Update UI with progress
  },
  error: (err) => {
    console.error('Subscription error:', err);
    // Handle connection errors
  },
  complete: () => {
    console.log('Subscription completed');
    // Clean up resources
  }
});

// Clean up subscription when component unmounts
subscription.unsubscribe();
```

## Security Considerations

### 1. Query Depth Limiting

The API implements query depth limiting to prevent deeply nested queries that could cause performance issues.

### 2. Query Complexity Analysis

Complex queries that request too many fields or nested relationships may be rejected.

### 3. Authentication Required

Most queries and mutations require valid authentication tokens.

### 4. Field-Level Permissions

Some fields are only accessible to users with appropriate permissions (e.g., admin-only system metrics).

## Integration Examples

### JavaScript/TypeScript

```typescript
import { ApolloClient, InMemoryCache, gql, createHttpLink } from '@apollo/client';
import { setContext } from '@apollo/client/link/context';

// Create HTTP link
const httpLink = createHttpLink({
  uri: 'http://localhost:8000/graphql',
});

// Add auth header
const authLink = setContext((_, { headers }) => {
  const token = localStorage.getItem('access_token');
  return {
    headers: {
      ...headers,
      authorization: token ? `Bearer ${token}` : "",
    }
  }
});

// Create Apollo client
const client = new ApolloClient({
  link: authLink.concat(httpLink),
  cache: new InMemoryCache()
});

// Example query
const GET_TRANSCRIPTION_JOBS = gql`
  query GetTranscriptionJobs($page: Int!, $perPage: Int!) {
    transcriptionJobs(
      pagination: { page: $page, perPage: $perPage }
    ) {
      jobs {
        id
        status
        filename
        createdAt
        transcript
        confidence
      }
      totalCount
      hasNext
    }
  }
`;

// Execute query
client.query({
  query: GET_TRANSCRIPTION_JOBS,
  variables: { page: 1, perPage: 10 }
}).then(result => {
  console.log('Transcription jobs:', result.data.transcriptionJobs);
});
```

### Python

```python
import requests
import json

# GraphQL endpoint
url = "http://localhost:8000/graphql"

# Headers with authentication
headers = {
    "Authorization": "Bearer your_access_token",
    "Content-Type": "application/json"
}

# Query
query = """
query {
  currentUser {
    id
    email
    fullName
    creditsRemaining
  }
}
"""

# Execute query
response = requests.post(
    url, 
    json={"query": query}, 
    headers=headers
)

if response.status_code == 200:
    data = response.json()
    if "errors" not in data:
        user = data["data"]["currentUser"]
        print(f"User: {user['fullName']} ({user['email']})")
        print(f"Credits remaining: {user['creditsRemaining']}")
    else:
        print("GraphQL errors:", data["errors"])
else:
    print(f"HTTP error: {response.status_code}")
```

### cURL

```bash
# Query example
curl -X POST http://localhost:8000/graphql \
  -H "Authorization: Bearer your_access_token" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query { currentUser { id email fullName creditsRemaining } }"
  }'

# Mutation example
curl -X POST http://localhost:8000/graphql \
  -H "Authorization: Bearer your_access_token" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mutation($input: UserLoginInput!) { loginUser(input: $input) { ... on AuthenticationResult { success accessToken user { email } } } }",
    "variables": {
      "input": {
        "email": "user@example.com",
        "password": "password123"
      }
    }
  }'
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure Bearer token is included in Authorization header
   - Check token expiration and refresh if needed

2. **Query Errors**
   - Validate GraphQL syntax using the playground
   - Check field names and types against schema

3. **Rate Limiting**
   - Monitor rate limit headers
   - Implement exponential backoff for retries

4. **File Upload Issues**
   - Use multipart form data for file uploads
   - Check file size limits (100MB max)

5. **Subscription Connection Issues**
   - Verify WebSocket connection is established
   - Check authentication for subscription endpoints

### Getting Help

- Use the GraphQL playground for interactive debugging
- Check server logs for detailed error information
- Review this documentation for proper usage patterns
- Contact support for API-specific issues

---

For more information, see:
- [REST API Documentation](./api.md)
- [Authentication Guide](./authentication.md)
- [SDKs and Client Libraries](./sdks.md)