package com.egoadl.egomotion;

import androidx.appcompat.app.AppCompatActivity;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.media.MediaScannerConnection;
import android.media.ToneGenerator;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.os.PowerManager;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemSelectedListener;
import android.widget.Button;
import android.hardware.SensorEventListener;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.ArrayAdapter;
import android.widget.Toast;

import org.w3c.dom.Text;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.DatagramSocket;
import java.net.Socket;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;

public class MainActivity extends AppCompatActivity {
    private Button btnDataCollect, btnDataCheck;
    private TextView textPacket;
    private EditText textTime, textNum, textChannel;
    private SensorManager mSensorManager;
    private Sensor mAccelerometer, mGyrosocpe;
    private TestSensorListener mSensorListener;
    private TestSensorListener myService;
    private boolean blnRecord = false;
    private boolean Acclog = false;
    private boolean Gyrolog = false;

    private AudioRecord audioRecord;
    private int channelConfig = AudioFormat.CHANNEL_IN_MONO;
    private int encodingBitrate = AudioFormat.ENCODING_PCM_16BIT;

    private int recBufSize = 0;
    private int frameSize = 3072;
    private int sampleRateInHz = 48000;
    private int Acc_sampleRateInHz = 250;
    private int Gyro_sampleRateInHz = 250;

    private int timelen = 60*10;//60*10;
    private int test_timelen = 3;
    private int packet_num = 0;
    private Queue<Double> AccBuffer_data = new ArrayDeque<>(Acc_sampleRateInHz * (timelen+1) * 3);
    private Queue<Long> Acc_time_stamp = new ArrayDeque<>(Acc_sampleRateInHz * (timelen+1));
    private long[] Acc_time_stamp_buffer = new long[Acc_sampleRateInHz * (timelen+1)];
    private double[] acc_buffer = new double[Acc_sampleRateInHz * (timelen+1) * 3];

    private Queue<Double> GyroBuffer_data = new ArrayDeque<>(Gyro_sampleRateInHz * (timelen+1) * 3);
    private Queue<Long> Gyro_time_stamp = new ArrayDeque<>(Gyro_sampleRateInHz * (timelen+1));
    private long[] Gyro_time_stamp_buffer = new long[Gyro_sampleRateInHz * (timelen+1)];
    private double[] gyro_buffer = new double[Gyro_sampleRateInHz * (timelen+1) * 3];
//    private byte[] save_recording = new byte[sampleRateInHz * 2 * (timelen+1)];

    private boolean sendDatatoFile = true;
    private PowerManager.WakeLock mWakeLock;

    private int count = 0;
    private int count_max = 3;

    //channel
    String[] channels = { "Ku", "Ou",
            "au", "m+"};


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        btnDataCollect = (Button)findViewById(R.id.button);
        btnDataCheck = (Button)findViewById(R.id.buttoncheck);


        textPacket = (TextView)findViewById(R.id.textView);
        textTime = (EditText) findViewById(R.id.edittexttime);
        textNum = (EditText) findViewById(R.id.edittextnum);
        textChannel = (EditText) findViewById(R.id.edittextchannel);
        mSensorListener = new TestSensorListener();
        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mGyrosocpe = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        myService = new TestSensorListener();
        PowerManager manager = (PowerManager) getSystemService(Context.POWER_SERVICE);
        mWakeLock = manager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Egomotion:ABC");
        IntentFilter filter = new IntentFilter(Intent.ACTION_SCREEN_ON);
        filter.addAction(Intent.ACTION_SCREEN_OFF);
        registerReceiver(myService.mReceiver, filter);

        // Test the data
        btnDataCollect.setEnabled(false);
//        textPacket.setText("Testing the WiFi CSI data collecting...");
//        ThreadInstantWiFi ThreadWiFi = new ThreadInstantWiFi();
//        ThreadWiFi.setTimelen(test_timelen * -1);
//        String filename = Environment.getExternalStorageDirectory() + "/Download/";
//        filename += "test";
//        ThreadWiFi.setFilename(filename);
//        ThreadWiFi.start();

        btnDataCheck.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View v)
            {
                btnDataCheck.setEnabled(false);
                textPacket.setText("Testing the WiFi CSI data collecting...");
                ThreadInstantWiFi ThreadWiFi = new ThreadInstantWiFi();
                ThreadWiFi.setTimelen(test_timelen * -1);
                String filename = Environment.getExternalStorageDirectory() + "/Download/";
                filename += "test";
                ThreadWiFi.setFilename(filename);
                ThreadWiFi.start();
            }

        });

        btnDataCollect.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View v)
            {
                recBufSize = AudioRecord.getMinBufferSize(sampleRateInHz,
                        channelConfig, encodingBitrate);

                audioRecord = new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,
                        sampleRateInHz, channelConfig, encodingBitrate, recBufSize);
//                audioRecord = new AudioRecord(MediaRecorder.AudioSource.VOICE_COMMUNICATION,
//                        sampleRateInHz, channelConfig, encodingBitrate, recBufSize);
                timelen = Integer.valueOf(textTime.getText().toString());
                count_max = Integer.valueOf(textNum.getText().toString());
                count = 0;
                new_run();

//                String filename = Environment.getExternalStorageDirectory() + "/Download/";
//                String filefolder = Environment.getExternalStorageDirectory() + "/Download";
//                createFile(filefolder);
//                filename += String.valueOf(System.currentTimeMillis());
//                mWakeLock.acquire();
//                btnDataCollect.setEnabled(false);
//                ThreadInstantWiFi ThreadWiFi = new ThreadInstantWiFi();
//                ThreadWiFi.setFilename(filename);
//                ThreadInstantRecord ThreadRecord = new ThreadInstantRecord();
//                ThreadRecord.setFilename(filename);
//
//                ThreadWiFi.start();
//                ThreadRecord.start();
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(mSensorListener, mAccelerometer, 2000);
        mSensorManager.registerListener(mSensorListener, mGyrosocpe, 2000);
        //mSensorManager.registerListener(mSensorListener, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
    }

    @Override
    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(mSensorListener);
    }

    private Handler mHandler = new Handler(){
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            blnRecord=false;
            // textview_status.setText("Finish Recording...");
            packet_num = msg.getData().getInt("packet_number");
            textPacket.setText("Packet Number: " + String.valueOf(packet_num) + ", Data Number:" + String.valueOf(count));
            Log.i("Packet_number:", String.valueOf(packet_num));
        }
    };

    private Handler testHandler = new Handler(){
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            // textview_status.setText("Finish Recording...");
            packet_num = msg.getData().getInt("packet_number");
            Log.i("Packet_number:", String.valueOf(packet_num));
//            !!!
//            if (packet_num / test_timelen > 200) {
            if (packet_num / test_timelen >= 0) {
                textPacket.setText("WiFi CSI data collecting testing pass!");
                btnDataCollect.setEnabled(true);
                btnDataCheck.setEnabled(false);
                packet_num = 0;
            }
            else {
                textPacket.setText("WiFi CSI data collecting testing fail!");
                btnDataCheck.setEnabled(true);
            }
        }
    };

    private void new_run(){
        textPacket.setText("Packet Number: " + String.valueOf(packet_num) + ", Data Number:" + String.valueOf(count));
        count += 1;
        btnDataCollect.setEnabled(false);

        String filename = Environment.getExternalStorageDirectory() + "/Download/";
        String filefolder = Environment.getExternalStorageDirectory() + "/Download";
        createFile(filefolder);
        filename += String.valueOf(System.currentTimeMillis());
        mWakeLock.acquire();
        ThreadInstantWiFi ThreadWiFi = new ThreadInstantWiFi();
        ThreadWiFi.setFilename(filename);
        ThreadWiFi.setTimelen(timelen);
        ThreadInstantRecord ThreadRecord = new ThreadInstantRecord();
        ThreadRecord.setFilename(filename);

        ThreadWiFi.start();
        ThreadRecord.start();
    }

    private Handler wHandler = new Handler(){
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            btnDataCollect.setEnabled(true);
            if (count<count_max && btnDataCollect.isEnabled()!=false){
                new_run();
            }
            else {
                if (count_max != 1) {
                    ToneGenerator toneG = new ToneGenerator(AudioManager.STREAM_ALARM, 80);
                    toneG.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 1000);
                }
            }
        }
    };



    class ThreadInstantWiFi extends Thread
    {
        private String filename;
        private int timelen;
        private int channel;

        public void setFilename(String filename) {
            this.filename = filename;
        }
        public void setTimelen(int timelen) {
            this.timelen = timelen;
        }
        @Override
        public void run()
        {
            channel = Integer.valueOf(textChannel.getText().toString());
            String channel_str = "au";
            switch (channel){
                case 36: channel_str = "Ku";
                         break;
                case 52: channel_str = "Ou";
                         break;
                case 100: channel_str = "au";
                        break;
                case 149: channel_str = "m+";
                        break;
            }
            Log.i("channel", channel_str);

            //String spec = channel_str + "ABEQAAAQA8fD9iJiQAAAAAAAAAAAAAAAAAAAAAAAAAAA==";
//!!!
//            String spec = channel_str + "ABEQAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAAAA==";
//            Oculus MAC ADDRESS
            String spec = channel_str +  "IBEQAAAQCA8+811Z4AAAAAAAAAAAAAAAAAAAAAAAAAAA==";
            sudoForResult("ifconfig wlan0 up");
//            Channel 146
//            sudoForResult("nexutil -Iwlan0 -s500 -b -l34 -vm+ABEQAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAAAA==");
//            Channel 100

//            -c 100/80 -C 1 -N 1
//            auABEQAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAAAA==
//                    -c 100/40 -C 1 -N 1
//            ZtgBEQAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAAAA==
//                    -c 100/20 -C 1 -N 1
//            ZNABEQAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAAAA==
//                    -c 149/80 -C 1 -N 1
//            m+ABEQAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAAAA==
//                    -c 149/40 -C 1 -N 1
//            l9gBEQAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAAAA==
//                    -c 149/20 -C 1 -N 1
//            ldABEQAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAAAA==
//                    -c 100/80 -C 1 -N 0x7
//            auABcQAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAAAA==
//                    -c 100/40 -C 1 -N 0x7
//            ZtgBcQAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAAAA==
//                    -c 100/20 -C 1 -N 0x7
//            ZNABcQAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAAAA==
//                    -c 100/80 -C 1 -N 0xf
//            auAB8QAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAAAA==
//                    -c 100/40 -C 1 -N 0xf
//            ZtgB8QAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAAAA==
//                    -c 100/20 -C 1 -N 0xf
//            ZNAB8QAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAAAA==
//                    -c 100/80 -C 1 -N 1 -d 10
//            auABEQAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAKAA==
//                    -c 100/80 -C 1 -N 1 -d 100
//            auABEQAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAABkAA==
//                    -c 100/80 -C 1 -N 0xf -d 100
//            auAB8QAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAABkAA==
            // router 1:auABEQAAAQAkS/6/FMQAAAAAAAAAAAAAAAAAAAAAAAAAAA==
            // router 2:auABEQAAAQA8fD9iJiAAAAAAAAAAAAAAAAAAAAAAAAABAA==
            sudoForResult("nexutil -Iwlan0 -s500 -b -l34 -v" + spec);
            sudoForResult("nexutil -Iwlan0 -m1");
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            Log.i("Save1:Startcsi",Long.toString(System.currentTimeMillis()));
            blnRecord = true;
            long time_start = System.currentTimeMillis();
            sudoForResult("timeout " + String.valueOf(Math.abs(this.timelen)) + " tcpdump -i wlan0 dst port 5500 -w " + this.filename + ".pcap");
            long time_end = System.currentTimeMillis();
            Log.i("Save1:Stopcsi",Long.toString(System.currentTimeMillis()));
            File mFile = new File(this.filename + ".pcap");
            try {
                long size = getFileSize(mFile);
                double packet_num = size/1100.0;
                // Log.d("Results:", "### size: " + size);

                Message msg = new Message();
                msg.what = 1;
                Bundle bundle = new Bundle();
                bundle.putInt("packet_number", (int)packet_num);
                Log.i("Save:Packet",Integer.toString((int)packet_num));
                Log.i("Save:Time",Long.toString(time_end - time_start));
                msg.setData(bundle);
                if (this.timelen > 0) {
                    MediaScannerConnection.scanFile(getApplicationContext(), new String[]{mFile.getAbsolutePath()}, null, null);
                    mHandler.sendMessage(msg);
                }
                else {
                    testHandler.sendMessage(msg);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    class ThreadInstantRecord extends Thread {
        private String filename;

        public void setFilename(String filename) {
            this.filename = filename;
        }
        @Override
        public void run() {
            short[] bsRecord = new short[recBufSize * 2];
            byte[] save_recording = new byte[sampleRateInHz * 2 * (timelen+1)];
            int datacount = 0;

            String accfilecsv = filename + "_acc.csv";
            String gyrofilecsv = filename + "_gyro.csv";
            filename += ".wav";

            WavFileWriter filewriter = new WavFileWriter();
            int now =0;

            while (blnRecord == false) {}
            Log.i("Save1:Startrecording",Long.toString(System.currentTimeMillis()));
            audioRecord.startRecording();
            Acclog = true;
            Gyrolog = true;

            while (blnRecord) {
                int line = 0;
                if (encodingBitrate == AudioFormat.ENCODING_PCM_16BIT) {
                    line = audioRecord.read(bsRecord, 0, frameSize * 2); }

                datacount = datacount + line;
                if (line >= frameSize) {
                    now++;
                    Log.i("Save:record", Integer.toString(now));
                    //mylog("Frame length:" + line);
                    if (sendDatatoFile == true) {
                        int j=(datacount - line)*2;
                        if (encodingBitrate == AudioFormat.ENCODING_PCM_16BIT) {
                            for (int i = 0; i < line; i++) {
                                save_recording[j++] = (byte) (bsRecord[i] & 0xFF);
                                save_recording[j++] = (byte) (bsRecord[i] >> 8);
                            }
                        }
                    }
                }
                //mylog("datacount: "+datacount);
                if (datacount > timelen * sampleRateInHz) {
                    break;
                }
            }
            Log.i("Save1:Stoprecording",Long.toString(System.currentTimeMillis()));
            Log.i("Save:wavtofile","successfully" + String.valueOf(blnRecord));
            Acclog = false;
            Gyrolog = false;
            audioRecord.stop();

            // Acc data logging
            int i=0;
            for (Long index: Acc_time_stamp) {
                Acc_time_stamp_buffer[i] = index.longValue();
                i++;
            }
            i=0;
            for (Double index: AccBuffer_data) {
                acc_buffer[i] = index.doubleValue();
                i++;
            }
            //mylog("Timenow:" + (time_stamp_buffer[0] - time_stamp_buffer[Time_stamp.size()-1]));
            List<String> header = new ArrayList<>();
            header.add("Time");
            header.add("X");
            header.add("Y");
            header.add("Z");

            List<List<String>> contents = new ArrayList<>();
            for(i = 0 ; i < Acc_time_stamp.size() ; i++)
            {
                List<String> oneLineColumns = new ArrayList<>();
                oneLineColumns.add(String.valueOf(Acc_time_stamp_buffer[i]));
                oneLineColumns.add(String.valueOf(acc_buffer[i*3+0]));
                oneLineColumns.add(String.valueOf(acc_buffer[i*3+1]));
                oneLineColumns.add(String.valueOf(acc_buffer[i*3+2]));
                contents.add(oneLineColumns);
            }
            int index_now = Acc_time_stamp.size()-1;
            AccBuffer_data.clear();
            Acc_time_stamp.clear();

            //Gyro data logging
            i=0;
            for (Long index: Gyro_time_stamp) {
                Gyro_time_stamp_buffer[i] = index.longValue();
                i++;
            }
            i=0;
            for (Double index: GyroBuffer_data) {
                gyro_buffer[i] = index.doubleValue();
                i++;
            }
            //mylog("Timenow:" + (time_stamp_buffer[0] - time_stamp_buffer[Time_stamp.size()-1]));
//            List<String> gyro_header = new ArrayList<>();
//            header.add("Time");
//            header.add("X");
//            header.add("Y");
//            header.add("Z");


            List<List<String>> gyro_contents = new ArrayList<>();
            for(i = 0 ; i < Gyro_time_stamp.size() ; i++)
            {
                List<String> oneLineColumns = new ArrayList<>();
                oneLineColumns.add(String.valueOf(Gyro_time_stamp_buffer[i]));
                oneLineColumns.add(String.valueOf(gyro_buffer[i*3+0]));
                oneLineColumns.add(String.valueOf(gyro_buffer[i*3+1]));
                oneLineColumns.add(String.valueOf(gyro_buffer[i*3+2]));
                gyro_contents.add(oneLineColumns);
            }
//            index_now = Gyro_time_stamp.size()-1;
            GyroBuffer_data.clear();
            Gyro_time_stamp.clear();

            Log.i("Save:imucsv",Long.toString(Acc_time_stamp_buffer[index_now] - Acc_time_stamp_buffer[0]));
            if (count_max == 1) {
                ToneGenerator toneG = new ToneGenerator(AudioManager.STREAM_ALARM, 80);
                toneG.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 1000);
                Log.i("Save1:StopTone",Long.toString(System.currentTimeMillis()));
            }
            if (Acc_time_stamp_buffer[index_now] - Acc_time_stamp_buffer[0] > (timelen-1) * 1000) {
                CsvWriter.toCsvFile(header, contents, accfilecsv);
                CsvWriter.toCsvFile(header, gyro_contents, gyrofilecsv);

                // write wav
                try {
                    filewriter.openFile(filename, sampleRateInHz, 16, 1);
                    filewriter.writeData(save_recording, 0, timelen * sampleRateInHz * 2);
                    filewriter.closeFile();
                    MediaScannerConnection.scanFile(getApplicationContext(), new String[] {accfilecsv, gyrofilecsv, filename}, null, null);
                    Log.i("Save:wavtofile",Long.toString(System.currentTimeMillis() - Acc_time_stamp_buffer[0]));

                } catch (IOException ie) {
                    ie.printStackTrace();
                }
            }
            wHandler.sendEmptyMessage(0);

        }
    }

    public static String sudoForResult(String...strings) {
        String res = "";
        DataOutputStream outputStream = null;
        InputStream response = null;
        try{
            Process su = Runtime.getRuntime().exec("su");
            outputStream = new DataOutputStream(su.getOutputStream());
            response = su.getInputStream();

            for (String s : strings) {
                outputStream.writeBytes(s+"\n");
                outputStream.flush();
            }

            outputStream.writeBytes("exit\n");
            outputStream.flush();
            try {
                su.waitFor();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            res = readFully(response);
            Log.i("aaa:", res);
        } catch (IOException e){
            Log.i("Error", String.valueOf(e));
            e.printStackTrace();
        } finally {
            Closer.closeSilently(outputStream, response);
        }
        return res;
    }
    public static String readFully(InputStream is) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        int length = 0;
        while ((length = is.read(buffer)) != -1) {
            baos.write(buffer, 0, length);
        }
        return baos.toString("UTF-8");
    }
    public static long getFileSize(File file) throws Exception {
        if (file == null) {
            return 0;
        }
        long size = 0;
        if (file.exists()) {
            FileInputStream fis = null;
            fis = new FileInputStream(file);
            size = fis.available();
        }
        return size;
    }

    public void createFile(String fileName) {
        File file = new File(fileName);
        if (fileName.indexOf(".") != -1) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        } else {
            file.mkdir();
        }
    }


    private class TestSensorListener implements SensorEventListener {

        public BroadcastReceiver mReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {

                if (!blnRecord
                        && !intent.getAction().equals(Intent.ACTION_SCREEN_OFF)) {
                    return;
                }
                if (mSensorManager != null) {
                    mSensorManager.unregisterListener(TestSensorListener.this);
                    mSensorManager
                            .registerListener(
                                    TestSensorListener.this,
                                    mSensorManager
                                            .getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
                                    2000);
                    mSensorManager
                            .registerListener(
                                    TestSensorListener.this,
                                    mSensorManager
                                            .getDefaultSensor(Sensor.TYPE_GYROSCOPE),
                                    2000);
                }

            }

        };

        @Override
        public void onSensorChanged(SensorEvent event) {
//            Log.i("Acc:", String.valueOf(event.values[1]));
            Sensor source = event.sensor;
            if (Acclog && source.equals(mAccelerometer)) {
                //Log.i("aaa:", "bbb");
                Acc_time_stamp.add(System.currentTimeMillis());
                AccBuffer_data.add(new Double(event.values[0]));
                AccBuffer_data.add(new Double(event.values[1]));
                AccBuffer_data.add(new Double(event.values[2]));
            }
            if (Gyrolog && source.equals(mGyrosocpe)) {
//                Log.i("aaa:", "bbb");
                Gyro_time_stamp.add(System.currentTimeMillis());
                GyroBuffer_data.add(new Double(event.values[0]));
                GyroBuffer_data.add(new Double(event.values[1]));
                GyroBuffer_data.add(new Double(event.values[2]));
            }

        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {
        }

    }
}
