# Dataset Download Instruction

Our dataset is hosted on two sites:

- [KAUST Library (Official)](https://repository.kaust.edu.sa/items/581bd796-dad1-4b44-9fc1-9d1f584e100f)
- [Hugging Face (Mirror)](https://huggingface.co/datasets/cocoakang/CoralSpec-30M)

## Download From KAUST Library

The dataset is hosted on KAUST library and can be downloaded through Globus. Globus is a reliable data transfer service that creates connection between your local machine and the KAUST server.

#### Step 1: Open the KAUST dataset page and click the Globus download link

On the KAUST page, click **Link to Globus download directory**.

![Step 1 - KAUST dataset page](figures/step_1.jpg)

#### Step 2: Log in to Globus

If it is the first time you open Globus, you will need to sign in. Here we use google account for demonstration. The dataset has been set to public access.

![Step 2 - Globus login](figures/step_2_login.jpg)

#### Step 3 (through web browser): Download data (1 file at a time)

Users can directly download one file through the browser, shown as below. However, this method is not recommended for large files or multiple files.

![Step 3 - Download](figures/step_4_download_website.jpg)

Sometimes Globus asks for an additional identity check for downloading. Please login use the same account to proceed.

![Step 3b - Additional login required](figures/step_5_additional_login.jpg)

After login, you will be redirected to the file download page. 

![Step 3c - File download page](figures/step_6_redirect.jpg)

**Warning 1**: Sometimes due to network issues, the download may fail and pops up an error message as below. In this case, please use Globus Personal Connect for more reliable transfer.

![Step 3d - Download error](figures/internal_server_error.jpg)

**Warning 2**: This method can only download 1 file at a time. If you check multiple files, then the "Download" button will be disabled. For downloading multiple files, please use Globus Personal Connect.

![Step 3e - Multiple file download disabled](figures/step_7_double_column.jpg)

#### Step 3 (through Globus Personal Connect): Set up Globus Personal Connect

Globus Connect Personal is an official client application to establish connection with your machine and KAUST library and transfer data. Please download and install it on your local machine [here](https://www.globus.org/globus-connect-personal). 

After installation, please login to Globus Connect Personal with the same account you used for the web login. 

![Step 3 - Download](figures/step_8_globus_login.jpg)
![Step 3b - Globus login](figures/step_9_globus_login_auth.jpg)

After login, you will be asked to set up your local collection. Please provide a name for the collection. 
The default download path is set to home path for the user. You can change it to any folder you want later.  

![Step 3c - Local collection setup](figures/step_10_globus_local_setting.jpg)

When the setup is successful, you should see the following message. Keep Globus Connect Personal running to maintain the connection.

![Step 3d - Setup successful](figures/step_11_globus_set_done.jpg)


#### Step 4: Find your local collection in Globus web app

Go back to the Globus web app, open collection search, and find your local collection (for example `my_mac`).

![Step 4 - Double Column view](figures/step_7_double_column.jpg)

![Step 4 - Find my collection](figures/step_12_globus_web_search.jpg)

![Step 4b - Select my collection](figures/step_13_globus_my_collection.jpg)

![Step 4c - Collection details](figures/step_14_setup_local_folder.jpg)

#### Step 5: Select files and start transfer

Select one or more dataset zip files, then click **Start**.

![Step 5 - Select files and start](figures/step_15_select_and_download.jpg)

You should see a transfer request submitted message.

![Step 5b - Transfer submitted](figures/step_16_start_session_successfully.jpg)

#### Step 8: Monitor progress in Activity

Open **Activity** to monitor running tasks.

![Step 8 - Activity list](figures/step_17_activity.jpg)

Click the task to view details and transfer statistics.

![Step 8b - Activity details](figures/step_18_downloading.jpg)


## Download From Hugging Face

The dataset is also mirrored on Hugging Face. You can download the dataset through Hugging Face's web interface.

Once open the Hugging Face dataset page, click the "Files and versions" tab, then click the download icon to download the zip files.

![Hugging Face download](figures/hugging_face.jpg)
