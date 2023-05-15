package com.kesun.egomotion;

import android.os.Environment;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Created by jason on 2017/12/8.
 */

public class CsvWriter {

    private static final String CSV_FILE_PATH = Environment.getExternalStorageDirectory() + File.separator + "download" + File.separator;

    private static final String CSV_FILE__NAME_EXTENSION = ".csv";

    private static final String CSV_FILE_CHARSET = "UTF-8";

    private static final byte[] CSV_FILE_BOM = new byte[]{(byte) 0xEF, (byte) 0xBB, (byte) 0xBF};

    private static final String CSV_SEPARATOR = ",";


    public static void toCsvFile(List<String> title, List<List<String>> contents, String fileName) {
        try
        {
            File file = createFile(fileName);

            FileOutputStream fos = new FileOutputStream(file);
            fos.write(CSV_FILE_BOM);

            OutputStreamWriter osw = new OutputStreamWriter(fos, CSV_FILE_CHARSET);
            BufferedWriter bw = new BufferedWriter(osw);

            writeOneLine(bw, title);
            for(List<String> oneLine : contents)
                writeOneLine(bw, oneLine);

            bw.flush();
            bw.close();
        }
        catch (UnsupportedEncodingException e) { }
        catch (FileNotFoundException e) { }
        catch (IOException e) { }
    }


    private static File createFile(String fileName) throws IOException {
        //String fileName = new SimpleDateFormat("yyyyMMddHHmmss", Locale.US).format(new Date());
        File file = new File(fileName);
        if (!file.exists())
            file.createNewFile();
        return file;
    }


    private static void writeOneLine(BufferedWriter writer, List<String> oneLineColumns) throws IOException {
        if (null != writer && null != oneLineColumns)
        {
            StringBuffer oneLine = new StringBuffer();
            for (String column : oneLineColumns)
            {
                oneLine.append(CSV_SEPARATOR);
                oneLine.append("\"");
                oneLine.append(null != column ? column.replaceAll("\"", "\"\"") : "");
                oneLine.append("\"");
            }

            writer.write(oneLine.toString().replaceFirst(",", ""));
            writer.newLine();
        }
    }

}