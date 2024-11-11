# Connecting Code to Local Google Drive

This guide will walk you through setting up Google Drive on your desktop to save and access project files locally.

## Steps

1. **Download Google Drive for Desktop**  
   Download and install [Google Drive for Desktop]([https://www.google.com/drive/download/](https://support.google.com/drive/answer/10838124?hl=en)).

2. **Log in to Your Google Account**  
   Open Google Drive for Desktop and sign in to your Google account.

3. **Access Google Drive as a Drive on Your Computer**  
   Once logged in, Google Drive will appear as a local drive on your computer (e.g., `G:\` on Windows or `/Volumes/GoogleDrive` on macOS).

4. **Locate the "RL Project (Fall 2024)" Folder in Google Drive**  
   If you do not see the `"RL Project (Fall 2024)"` folder under **My Drive**:
   
   - Go to **Shared with me** in Google Drive.
   - Find the `"RL Project (Fall 2024)"` folder.
   - Click on the three dots next to the folder name, select **Organize**, and choose **Add shortcut to Drive**. Then, select **My Drive**.

5. **Set the Path to Google Drive in the Code**  
   In your code, set the `path_to_google_drive` variable to the correct local path for your Google Drive. For example:
   
   ```python
   path_to_google_drive = "/path/to/Google Drive/My Drive/RL Project (Fall 2024)/Results/CartPole_Single"
