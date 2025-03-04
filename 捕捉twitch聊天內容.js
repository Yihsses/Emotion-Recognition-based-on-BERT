async function fetchTwitchComments(videoId, videoStart) {
    const url = 'https://gql.twitch.tv/gql';
    
    // 定義 GraphQL 查詢
    const query = {
        operationName: "VideoCommentsByOffsetOrCursor",
        variables: {
            videoID: videoId,
            contentOffsetSeconds: videoStart
        },
        extensions: {
            persistedQuery: {
                version: 1,
                sha256Hash: "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"
            }
        }
    };

    try {
        // 發送 POST 請求
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Client-ID': 'kd1unb4b3q4t58fwlpcbzcbnm76a8fp'  // 替換為你的 Client-ID
            },
            body: JSON.stringify(query)  // 將查詢轉換為 JSON 字符串
        });

        // 檢查請求是否成功
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        // 解析回應的 JSON 資料
        const data = await response.json();

        // 打印回應數據
        console.log(data);
        return data;  // 返回數據，這裡可以進一步處理評論數據

    } catch (error) {
        console.error('Error fetching Twitch comments:', error);
    }
}

// 範例調用
fetchTwitchComments('2396838102', 1975);